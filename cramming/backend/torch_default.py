"""Basic training backend for normal pytorch training."""
import torch

import os
import json
from omegaconf import OmegaConf
import logging
from functools import partial
import time

import transformers


from torch.distributed.optim import ZeroRedundancyOptimizer

from .utils import group_parameters, prepare_pretraining_dataloader, torchdynamo_compile_method, update_ema, updated_latest_weight_average
from .optimizers.schedulers import get_schedule_fn
from .optimizers import Adahessian, Shampoo, LARS, SAM, ProgressiveBatching

log = logging.getLogger(__name__)
_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)

import warnings

warnings.filterwarnings("ignore", "Detected call of ", UserWarning)  # Using a scheduler differently than pytorch


def initialize_torch(model, dataset, tokenizer, cfg_train, cfg_impl, setup=_default_setup):
    """initialize a torch engine."""
    if dataset is not None:
        dataloader = prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl)
    else:
        dataloader = None

    model_engine = TorchEngine(model, cfg_train, cfg_impl, setup=setup, seq_length=tokenizer.model_max_length)
    model_engine.train()
    return model_engine, model_engine.optimizer, model_engine.scheduler, dataloader


class TorchEngine(torch.nn.Module):
    """This class mirrors deepspeed functionality."""

    def __init__(self, model, cfg_train, cfg_impl, setup=_default_setup, seq_length=128):
        """Load Engine. This is the bare minimum init. The model is further traced if required."""
        super().__init__()

        self.cfg_train = cfg_train
        self.cfg_impl = cfg_impl
        if self.cfg_impl.microbatch_size is None:
            self.cfg_impl.microbatch_size = self.cfg_train.batch_size
        if self.cfg_impl.microbatch_size > self.cfg_train.batch_size:
            raise ValueError(f"MBS is {self.cfg_impl.microbatch_size}, but BS is only {self.cfg_train.batch_size}.")

        # Mixed Precision:
        enabled = self.cfg_impl.mixed_precision if setup["device"].type != "cpu" else False
        # Modules like LN are unsupported on CPU amp, so mixed precision args are disregarded on CPU
        # See https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior and check for layer_norm
        enable_scaling = self.cfg_impl.grad_scaling and self.cfg_impl.mixed_precision and setup["device"].type != "cpu"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaling)
        amp_dtype = getattr(torch, self.cfg_impl.mixed_precision_target_dtype) if setup["device"].type != "cpu" else torch.bfloat16
        self.amp_settings = dict(device_type=setup["device"].type, enabled=enabled, dtype=amp_dtype)

        # Optional batch size rampup
        self.current_batch_size = self.cfg_train.batch_size if self.cfg_train.batch_size_ramp == 0 else self.cfg_impl.microbatch_size

        # Microbatch accumulation
        self.accumulation_steps_expected = self.current_batch_size // self.cfg_impl.microbatch_size
        self.accumulated_samples = 0  # Record the number of samples seen, reset after triggering gradient update
        self.steps = 0  # Record the number of times "step" has been triggered

        # Optional sequence curriculum:
        self.sequence_curriculum = "sequence_curriculum" in cfg_train
        self.data_seq_length = seq_length
        self.current_seq_length = seq_length if not self.sequence_curriculum else cfg_train.sequence_curriculum.lengths[0]
        self.sequence_unfold = None if not self.sequence_curriculum else cfg_train.sequence_curriculum.unfold

        # Optional EMA/LAWA-type weight averages
        if "weight_averaging" in cfg_train:
            self.weight_averaging_frequency = cfg_train.weight_averaging.frequency
            self.weight_averaging = cfg_train.weight_averaging
            if self.weight_averaging.type == "EMA":
                self.param_store = [p.detach().clone() for p in model.parameters()]  # keep on CPU
                self.buffer_store = [b.detach().clone() for b in model.buffers()]
            else:
                self.store = []
        else:
            self.weight_averaging_frequency = 0

        # Choose setup and move model
        self.setup = setup
        model.to(**self.setup)

        # torchdynamo tracing?
        # Ideally would be able to compile the entire .step eventually
        self.forward = torchdynamo_compile_method(self.forward, cfg_impl.optimizer_context)

        # Old-school tracing?
        model = self._script_model(model)
        if torch.distributed.is_initialized():
            self.model = self._init_distributed(model)
        else:
            self.model = model
        try:
            self.forward_attention_masks = model.cfg.attention.causal_attention
        except (AttributeError, ValueError):
            self.forward_attention_masks = False

        self.optimizer, self.scheduler = _load_optimizer(model, cfg_train, cfg_impl)
        self.initial_time = time.time()

    def step(self, batch: dict[str, torch.Tensor]):
        loss = self.forward(**batch)["loss"]
        self.backward(loss)
        self.optimizer_step()
        return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids", "labels"]):
        """Move batch of data into device memory."""
        device_batch = {
            k: v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            for k, v in batch.items()
            if k in keys  # Add more keywords here if needed
        }
        self.set_sequence_curriculum_(device_batch)
        return device_batch

    def forward(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            return self.model(*inputs, **kwargs)

    def backward(self, loss):
        self.accumulated_samples += self.cfg_impl.microbatch_size
        return self.scaler.scale(loss / self.accumulation_steps_expected).backward()

    @torch.no_grad()
    def forward_inference(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            outputs = self.model(*inputs, **kwargs)["logits"]
        if outputs.shape[-1] == 1:
            predictions = outputs.squeeze(dim=-1)
        else:
            predictions = outputs.argmax(dim=-1)
        return outputs, predictions

    def optimizer_step(self):
        """Requires a scheduler that is based on iterations instead of epochs."""
        self.steps += 1
        if self.accumulated_samples >= self.current_batch_size:
            self.accumulated_samples = 0

            if self.cfg_train.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_train.gradient_clipping, norm_type=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.schedule_batch_size()
            self.schedule_curriculum()
            self.moving_average_computation()
        self.scheduler.step()  # Trigger in every step, otherwise things get annoying with grad accumulation

    def set_train_batch_size(self, batch_size):
        """Allow dynamic modifications of batch size."""
        self.current_batch_size = batch_size
        self.accumulation_steps_expected = self.current_batch_size // self.cfg_impl.microbatch_size

    def set_sequence_curriculum_(self, device_batch):
        """Assume huggingface data is B S"""
        if self.sequence_curriculum:
            for key, tensor in device_batch.items():
                if self.sequence_unfold:
                    device_batch[key] = tensor.view(-1, self.current_seq_length)
                else:
                    device_batch[key] = tensor[:, : self.current_seq_length].clone()

    def record_batch_size(self):
        if self.cfg_train.optim_mod.name != "progressive-batching":
            return self.current_batch_size
        else:
            return self.optimizer.last_full_step_accumulation * self.current_batch_size

    def record_tokens_per_step(self):
        """Tokens in each microbatch step."""
        if not self.sequence_curriculum:
            return self.current_seq_length * self.cfg_impl.microbatch_size
        else:
            if self.sequence_unfold:
                # Same number of tokens in this case:
                return self.current_seq_length * (self.data_seq_length // self.current_seq_length) * self.cfg_impl.microbatch_size
            else:
                # Reduced number of tokens here:
                return self.current_seq_length * self.cfg_impl.microbatch_size

    def schedule_batch_size(self):
        """Optionally implement linear batch size ramp-ups."""
        if (self.cfg_train.batch_size_ramp > 0) and (self.cfg_train.batch_size_ramp < 1):
            # interpret as percentage of total budget
            elapsed_hours = (time.time() - self.initial_time) / 60 / 60
            fake_step = int(elapsed_hours / self.cfg_train.budget * self.cfg_train.steps)

            batch_size_step = self.cfg_train.batch_size / (self.cfg_train.steps * self.cfg_train.batch_size_ramp)
            mbs = self.cfg_impl.microbatch_size
            new_batch_size = min(int(fake_step * batch_size_step // mbs + 1) * mbs, self.cfg_train.batch_size)
        elif self.steps < self.cfg_train.batch_size_ramp:
            batch_size_step = self.cfg_train.batch_size / self.cfg_train.batch_size_ramp
            # [(int(s * batch_size_step) // mbs + 1) * mbs for s in range(n) if s < ramp]
            mbs = self.cfg_impl.microbatch_size
            new_batch_size = int(self.steps * batch_size_step // mbs + 1) * mbs
        else:
            new_batch_size = self.cfg_train.batch_size
        self.set_train_batch_size(new_batch_size)

    def schedule_curriculum(self):
        """Optionally implement linear sequence lengths curriculum."""
        if self.sequence_curriculum:
            # Sequence curriculum should be a dict of two lists:
            # lengths (needs to be monotone ascending integers)
            # triggers (needs to be monotone ascending floats between 0 and 1)
            # and a keyword unfold = True/False
            elapsed_hours = (time.time() - self.initial_time) / 60 / 60
            fraction_elapsed = elapsed_hours / self.cfg_train.budget
            lengths = self.cfg_train.sequence_curriculum.lengths
            triggers = self.cfg_train.sequence_curriculum.triggers
            for trigger, length in zip(triggers, lengths):
                if fraction_elapsed > trigger:
                    self.current_seq_length = length

    def moving_average_computation(self):
        if self.weight_averaging_frequency > 0:
            if (self.steps % self.weight_averaging_frequency) == 0:
                params = [p.detach().cpu() for p in self.model.parameters()]
                buffers = [b.detach().cpu() for b in self.model.buffers()]
                if self.weight_averaging.type == "EMA":
                    update_ema(params, self.param_store, buffers, self.buffer_store, momentum=self.weight_averaging.momentum)
                else:  # latest weight averaging
                    self.param_store, self.buffer_store = updated_latest_weight_average(
                        params, buffers, self.store, last_k=self.weight_averaging.last_k
                    )

    @torch.no_grad()
    def retrieve_model_state_dict(self):
        if self.weight_averaging_frequency > 0:
            # Use weight averaged weights
            for param, param_ma in zip(self.model.parameters(), self.param_store):
                param.copy_(param_ma.data)
            for buffer, buffer_ma in zip(self.model.buffers(), self.buffer_store):
                buffer.copy_(buffer_ma.data)
            return self.model.state_dict()
        else:
            # Else use normal state dict
            return self.model.state_dict()

    def _init_distributed(self, model):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.setup["device"]] if self.setup["device"].type == "cuda" else None,
            output_device=self.setup["device"] if self.setup["device"].type == "cuda" else None,
            broadcast_buffers=self.cfg_impl.broadcast_buffers,
            bucket_cap_mb=self.cfg_impl.bucket_cap_mb,
            gradient_as_bucket_view=self.cfg_impl.gradient_as_bucket_view,
            static_graph=self.cfg_impl.static_graph,
        )
        return model

    def _script_model(self, model):
        if self.cfg_impl.jit == "trace":
            with torch.autocast(**self.amp_settings):
                # No guarantees for complicated models
                input_setup = dict(dtype=torch.long, device=self.setup["device"])
                templates = torch.randint(0, model.vocab_size, (*self.cfg_impl.trace_shape,), **input_setup)
                labels = torch.randint(0, model.vocab_size, (*self.cfg_impl.trace_shape,), **input_setup)

                model = torch.jit.trace(_ModelArgWrapper(model), (templates, labels), strict=False)
        elif self.cfg_impl.jit == "script":
            # This does not work for huggingface models
            model = torch.jit.script(model)
        return model

    def load_checkpoint(self, cfg_arch, file, skip_optim_state=True):
        """Load list of states from checkpoint file. Not generally compatible with any other engine?"""
        if file.startswith("hf://"):
            if file.endswith("-untrained"):
                log.info("Loading NO pretrained model as a sanity check ...")
            else:
                self.model = self.model.from_pretrained(file.split("hf://")[1], config=cfg_arch).to(**self.setup)
                # reinit optimizer:
                self.optimizer, self.scheduler = _load_optimizer(self.model, self.cfg_train, self.cfg_impl)
        else:
            if not skip_optim_state:
                optim_state, model_state, scheduler_state, _ = torch.load(file, map_location=self.setup["device"])
                self.model.load_state_dict(model_state).to(**self.setup)
                self.optimizer.load_state_dict(optim_state)
                self.scheduler.load_state_dict(scheduler_state)
            else:

                model_state = torch.load(file, map_location=self.setup["device"])
                try:
                    sanitized_state = {}
                    for k, v in model_state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        sanitized_state[k] = v
                    self.model.load_state_dict(sanitized_state, strict=True)
                except RuntimeError as e:
                    log.info(f"State dict difference is {str(e).split('Error(s) in loading state_dict for')[1]}... Ok?")
                    self.model.load_state_dict(sanitized_state, strict=False)
                self.model.to(**self.setup)

    def save_training_checkpoint(self, identifier, directory="checkpoints", state=None):
        """Path, identifier and additional client state. This checkpoint can be used to resume training.
        The default behavior is to save this checkpoint relative to the training working directory.
        """
        try:
            identifier_str = f"{identifier:2.4f}"
        except ValueError:
            identifier_str = str(identifier)
        file = os.path.join(directory, f"{identifier:2.4f}.pth")
        os.makedirs(directory, exist_ok=True)

        optim_state = self.optimizer.state_dict()
        model_state = self.retrieve_model_state_dict()
        scheduler_state = self.scheduler.state_dict()
        torch.save([optim_state, model_state, scheduler_state, state], file)

    def save_final_model(self, base_directory, identifier, tokenizer, cfg_arch, dryrun=False):
        """This checkpoint can be used for downstream tasks.
        The default behavior is to save this checkpoint to a checkpoints folder under base_directory/name/checkpoints"""
        try:
            identifier_str = f"{identifier:2.4f}"
        except ValueError:
            identifier_str = str(identifier)
        full_path = os.path.join(base_directory, "checkpoints", identifier_str)
        os.makedirs(full_path, exist_ok=True)
        # This saves tokenizer_config.json, tokenizer.json and special_tokens_map.json to this folder
        if not dryrun:
            tokenizer.save_pretrained(full_path)
            # Save model.pth, model_config.json
            torch.save(self.retrieve_model_state_dict(), os.path.join(full_path, "model.pth"))
            with open(os.path.join(full_path, "model_config.json"), "w") as file:
                json.dump(OmegaConf.to_container(cfg_arch, resolve=True), file)

    def push_to_hub(self, tokenizer, cfg, dryrun=False):
        """Analogous to save_final_model, but save model to hugginface hub."""
        from huggingface_hub import HfApi
        from io import BytesIO

        api = HfApi()

        if not dryrun:
            log.info(f"Pushing model to hub repository {cfg.impl.hf_directoy_name}.")
            final_state_dict = self.retrieve_model_state_dict()
            self.model.load_state_dict(final_state_dict)
            self.model.push_to_hub(cfg.impl.hf_directoy_name)
            tokenizer.push_to_hub(cfg.impl.hf_directoy_name)

            for config_group, config_name in zip([cfg.arch, cfg.data, cfg.train], ["arch", "data", "train"]):
                buffer = BytesIO()
                buffer.write(json.dumps(OmegaConf.to_container(config_group, resolve=True), indent=4).encode())
                api.upload_file(
                    path_or_fileobj=buffer,
                    path_in_repo=f"{config_name}_budget_hours_{cfg.budget}.json",
                    repo_id=f"{api.whoami()['name']}/{cfg.impl.hf_directoy_name}",  # there has to be a better way to do this, but ...
                    repo_type="model",
                )
        else:
            log.info(f"Skipping huggingface upload in dryrun state. Would upload to {cfg.impl.hf_directoy_name}.")

    def gradinit(self, data_iterable, optim_cfg, gradinit_cfg):
        """Run data-based initialization search as described in Zhu et al.,
        "GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training"

        Depends on functorch!

        This is gradinit without gradient aggregation, which allows higher-order derivatives
        """
        import functorch

        fmodel, params, buffers = functorch.make_functional_with_buffers(self.model)

        scales = [torch.tensor(1.0, **self.setup, requires_grad=True) for p in params]  # Modify all params by default
        # Prepare for functional optimizer:

        exp_avgs = [torch.tensor(0.0, **self.setup) for s in scales]
        exp_avg_sqs = [torch.tensor(0.0, **self.setup) for s in scales]
        state_steps = [torch.tensor(0.0, **self.setup) for s in scales]

        adam_fn = partial(torch.optim._functional.adam, amsgrad=False, beta1=0.9, beta2=0.98, weight_decay=0, eps=1e-6, maximize=False)

        eta = optim_cfg.lr
        for step in range(gradinit_cfg.steps):
            # scale params
            scaled_params = [p * s for p, s in zip(params, scales)]
            # ## Compute first step ##
            data_batch = self.to_device(next(data_iterable)[1])
            with torch.autocast(**self.amp_settings):
                loss0 = fmodel(**data_batch, params=scaled_params, buffers=buffers)["loss"]
            grads = torch.autograd.grad(loss0, scaled_params, create_graph=gradinit_cfg.second_order, only_inputs=True)
            gnorm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
            # Take first step
            # p <- p - eta*g
            if gradinit_cfg.step_type == "sign-grad":
                param_step = [p - eta * g.sign() for p, g in zip(scaled_params, grads)]
            elif gradinit_cfg.step_type == "norm-grad":
                param_step = [p - eta * g / gnorm for p, g in zip(scaled_params, grads)]
            else:
                param_step = [p - eta * g for p, g in zip(scaled_params, grads)]

            # ## Modify scales ##
            data_batch = self.to_device(next(data_iterable)[1])
            with torch.autocast(**self.amp_settings):
                loss1 = fmodel(**data_batch, params=param_step, buffers=buffers)["loss"]
            grads = torch.autograd.grad(loss1 / eta + (gnorm - 1).pow(2), scales, only_inputs=True)
            [g.zero_() for (name, _), g in zip(self.model.named_parameters(), grads) if "pos_embedding" in name]
            # Take adam step:
            with torch.no_grad():
                adam_fn(scales, grads, exp_avgs, exp_avg_sqs, [], state_steps, lr=gradinit_cfg.tau)
                # Project onto constraints and detach
                scales = [s.clamp_(min=gradinit_cfg.min_scale, max=gradinit_cfg.max_scale) for s in scales]
            log.info(f"Gradinit: Loss0: {loss0:2.4f}. Loss1: {loss1:2.4f}. Grad Norm: {gnorm:2.4f}.")
            # print([f"{name}:{s.item():2.4f}" for (name, _), s in zip(self.model.named_parameters(), scales)])

        # Finally copy scales into the existing model
        with torch.no_grad():
            for param, scale in zip(self.model.parameters(), scales):
                param.mul_(scale)


class _ModelArgWrapper(torch.nn.Module):
    """Wrap arguments."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels):
        return self.model(input_ids=input_ids, labels=labels)


def _load_optimizer(model, cfg_train, cfg_impl):

    # Filter some parameters
    grouped_parameters = group_parameters(model, cfg_train)

    # Select optimizer implementation
    if cfg_train.optim.type == "AdamW":
        optimizer_class = torch.optim.AdamW
    elif cfg_train.optim.type == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg_train.optim.type == "RAdam":
        optimizer_class = torch.optim.RAdam
    elif cfg_train.optim.type == "SGD":
        optimizer_class = torch.optim.SGD
    elif cfg_train.optim.type == "Adafactor":
        optimizer_class = transformers.Adafactor
    elif cfg_train.optim.type == "Shampoo":
        optimizer_class = Shampoo
    elif cfg_train.optim.type == "AdaHessian":
        optimizer_class = Adahessian
    elif cfg_train.optim.type == "AdamWScale":
        optimizer_class = AdamWScale
    elif cfg_train.optim.type == "Sophia-G":
        optimizer_class = Sophia
    elif cfg_train.optim.type == "Lion":
        from lion_pytorch import Lion

        optimizer_class = Lion

    elif cfg_train.optim.type == "Adam8bit":
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.Adam8bit
    elif cfg_train.optim.type == "AGD":
        depth = len(list(model.parameters()))
        optimizer_class = partial(AGD, depth=depth)
    else:
        raise ValueError(f"Invalid optimizer {cfg_train.optim.type} given.")
    optimizer_args = {k: v for k, v in cfg_train.optim.items() if k != "type"}
    if cfg_impl.foreach_optimizer and cfg_train.optim.type != "Shampoo":
        optimizer_args["foreach"] = True

    if torch.distributed.is_initialized() and cfg_impl.zero_redundancy_optimizer:
        # The overlap option is a whole bucket of problems in itself for now...
        optimizer = ZeroRedundancyOptimizer(
            grouped_parameters,
            optimizer_class=optimizer_class,
            parameters_as_bucket_view=True,
            overlap_with_ddp=False,
            **optimizer_args,
        )
    else:
        optimizer = optimizer_class(grouped_parameters, **optimizer_args)

    if cfg_train.optim_mod.name == "none":
        optimizer_to_schedule = optimizer
    else:
        optim_params = {k: v for k, v in cfg_train.optim_mod.items() if k != "name"}
        if cfg_train.optim_mod.name == "LARS":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "LARC":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "SAM":
            optimizer = SAM(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "progressive-batching":
            optimizer = ProgressiveBatching(optimizer, **optim_params)

        optimizer_to_schedule = optimizer.optim

    scheduler = get_schedule_fn(cfg_train)(optimizer_to_schedule)

    return optimizer, scheduler
