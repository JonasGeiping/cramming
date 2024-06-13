"""Basic training backend engine for pytorch training with all bells and whistles.

Interface set up to be compliant with the deepspeed engine interface.


There are two versions here, the TorchEngineMinimal, which is the default, and TorchEngineFull which contains a few training variations
that were tested but ultimately discarded, so read that part only if you're interested.

"""

import torch
import torch._inductor.utils

import os
import json
from omegaconf import OmegaConf
from functools import partial
from contextlib import nullcontext
import time

import logging

import transformers
from safetensors.torch import load_file, save_file
from transformers.utils.generic import working_or_temp_dir


from .utils import group_parameters, prepare_pretraining_dataloader, update_ema, updated_latest_weight_average
from .optimizers.schedulers import get_schedule_fn
from .optimizers import Adahessian, AdamWScale, Shampoo, LARS, SAM, ProgressiveBatching, AGD, Sophia

log = logging.getLogger(__name__)
_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)

import warnings

warnings.filterwarnings("ignore", "Detected call of ", UserWarning)  # schedulers are deliberately used differently


def initialize_torch(model, dataset, tokenizer, cfg_train, cfg_impl, elapsed_time, setup=_default_setup):
    """initialize a torch engine."""
    if dataset is not None:
        dataloader = prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl)
    else:
        dataloader = None

    # in most cases we can use a simpler Engine class:
    require_full_engine = "sequence_curriculum" in cfg_train or "weight_averaging" in cfg_train or "gradinit" in cfg_train

    if require_full_engine:
        model_engine = TorchEngineFull(model, cfg_train, cfg_impl, elapsed_time, setup=setup, seq_length=tokenizer.model_max_length)
    else:
        model_engine = TorchEngineMinimal(model, cfg_train, cfg_impl, elapsed_time, setup=setup, seq_length=tokenizer.model_max_length)
    model_engine.train()  # This is the default engine state. Pretraining scripts may change this.
    return model_engine, model_engine.optimizer, model_engine.scheduler, dataloader


class TorchEngineMinimal(torch.nn.Module):
    """This class mirrors deepspeed functionality. Not all changes are implemented in this version.

    See TorchEngineFull for more modifications.
    """

    def __init__(self, model, cfg_train, cfg_impl, already_elapsed_time=0.0, setup=_default_setup, seq_length=128):
        """Load Engine. The model will be compiled by default."""
        super().__init__()

        self.cfg_train = cfg_train
        self.cfg_impl = cfg_impl
        if self.cfg_impl.microbatch_size is None:
            self.cfg_impl.microbatch_size = self.cfg_train.batch_size
        if self.cfg_impl.microbatch_size > self.cfg_train.batch_size:
            raise ValueError(f"MBS is {self.cfg_impl.microbatch_size}, but BS is only {self.cfg_train.batch_size}.")
        self.current_seq_length = seq_length

        # Mixed Precision:
        enabled = self.cfg_impl.mixed_precision if setup["device"].type != "cpu" else False
        # Modules like LN are unsupported on CPU amp, so mixed precision args are disregarded on CPU
        # See https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior and check for layer_norm
        enable_scaling = self.cfg_impl.grad_scaling and self.cfg_impl.mixed_precision and setup["device"].type != "cpu"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaling)
        amp_dtype = getattr(torch, self.cfg_impl.mixed_precision_target_dtype) if setup["device"].type != "cpu" else torch.bfloat16
        self.amp_settings = dict(device_type=setup["device"].type, enabled=enabled, dtype=amp_dtype)

        # Choose setup and move model
        self.setup = setup
        model.to(**self.setup)

        from ..utils import flatten

        model = torch.compile(
            model,
            mode=self.cfg_impl.mode,
            dynamic=self.cfg_impl.dynamic,
            fullgraph=self.cfg_impl.fullgraph,
            backend=self.cfg_impl.backend,
            disable=not cfg_impl.compile_torch,
            # detailed options; cannot be given at the same time as mode:
            options=flatten(cfg_impl._inductor_vars, parent_key="", sep=".") if cfg_impl._inductor_vars is not None else None,
        )

        if torch.distributed.is_initialized():
            self.model = self._init_distributed(model)
            self.num_machines = torch.distributed.get_world_size()
        else:
            self.model = model
            self.model.no_sync = nullcontext
            self.num_machines = 1

        # Microbatch accumulation settings and counters
        self.effective_mbs = self.cfg_impl.microbatch_size * self.num_machines  # across machines
        self.current_batch_size = self.cfg_train.batch_size if self.cfg_train.batch_size_ramp == 0 else self.effective_mbs
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs
        self.accumulated_samples = 0  # Record the number of samples seen, reset after triggering gradient update
        self.steps = 0  # Record the number of times "step" has been triggered

        self.initial_time = time.time() - already_elapsed_time
        self.optimizer, self.scheduler = _load_optimizer(model, cfg_train, cfg_impl, self.initial_time)

    def step(self, batch: dict[str, torch.Tensor]):
        self.accumulated_samples += self.effective_mbs
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
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
        return device_batch

    def forward(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            return self.model(*inputs, **kwargs)

    def backward(self, loss):
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
        self.scheduler.step()  # Trigger in every step, otherwise things get annoying with grad accumulation

    def set_train_batch_size(self, batch_size):
        """Allow dynamic modifications of batch size."""
        self.current_batch_size = batch_size
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs

    def schedule_batch_size(self):
        """Optionally implement linear batch size ramp-ups."""
        if (self.cfg_train.batch_size_ramp > 0) and (self.cfg_train.batch_size_ramp < 1):
            # interpret batch_size_ramp as percentage of total budget:
            elapsed_hours = (time.time() - self.initial_time) / 60 / 60
            fake_step = int(elapsed_hours / self.cfg_train.budget * self.cfg_train.steps)

            batch_size_step = self.cfg_train.batch_size / (self.cfg_train.steps * self.cfg_train.batch_size_ramp)
            new_batch_size = min(int(fake_step * batch_size_step // self.effective_mbs + 1) * self.effective_mbs, self.cfg_train.batch_size)
        elif self.steps < self.cfg_train.batch_size_ramp:
            # interpret batch_size_ramp as fixed number of steps for ramp:
            batch_size_step = self.cfg_train.batch_size / self.cfg_train.batch_size_ramp
            new_batch_size = int(self.steps * batch_size_step // self.effective_mbs + 1) * self.effective_mbs
        else:
            new_batch_size = self.cfg_train.batch_size
        self.set_train_batch_size(new_batch_size)

    def record_batch_size(self):
        if self.cfg_train.optim_mod.name != "progressive-batching":
            return self.current_batch_size
        else:
            return self.optimizer.last_full_step_accumulation * self.current_batch_size

    def record_tokens_per_step(self):
        """Tokens in each microbatch step."""
        return self.current_seq_length * self.effective_mbs

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

    @torch.no_grad()
    def retrieve_model_state_dict(self):
        if self.cfg_impl.compile_torch:
            if torch.distributed.is_initialized():
                state_dict = self.model.module._orig_mod.state_dict()  # ughhhh
            else:
                state_dict = self.model._orig_mod.state_dict()  # ugh
        else:
            if torch.distributed.is_initialized():
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        return state_dict

    def load_checkpoint(self, cfg_arch, file, skip_optim_state=True):
        """Load list of states from checkpoint file. Not generally compatible with any other engine?"""
        if file.startswith("hf://"):
            if file.endswith("-untrained"):
                log.info("Loading NO pretrained model as a sanity check ...")
            else:
                self.model = self.model.from_pretrained(file.split("hf://")[1], config=cfg_arch).to(**self.setup)
                # reinit optimizer:
                self.optimizer, self.scheduler = _load_optimizer(self.model, self.cfg_train, self.cfg_impl, self.initial_time)
        else:
            model_state = load_file(file, device=str(self.setup["device"]))
            # This loader includes a few legacy options:
            if "encoder.embedding.word_embedding.weight" not in model_state:
                # Hack to save space when saving the model, more clever though would be save the right one in the first place
                model_state["encoder.embedding.word_embedding.weight"] = model_state["decoder.weight"]
            try:
                sanitized_state = {}
                for k, v in model_state.items():
                    if k.startswith("module."):
                        k = k[7:]
                    if self.cfg_impl.compile_torch:
                        k = f"_orig_mod.{k}"
                    if torch.distributed.is_initialized():
                        k = f"module.{k}"
                    sanitized_state[k] = v
                self.model.load_state_dict(sanitized_state, strict=True)
            except RuntimeError as e:
                log.info(f"State dict difference is {str(e).split('Error(s) in loading state_dict for')[1]}... Ok?")
                self.model.load_state_dict(sanitized_state, strict=False)
            self.model.to(**self.setup)

    def save_training_checkpoint(self, identifier="intermediate.pth", directory="", metadata=None):
        """Path, identifier and additional client state. This checkpoint can be used to resume training.
        The default behavior is to save this checkpoint relative to the training working directory.

        Has to be .pth because safetensors are annoying to dump a bunch of optim states, scales and schedules
        """
        file = os.path.join(directory, str(identifier))
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        save_state = dict()
        save_state["optim"] = self.optimizer.state_dict()
        save_state["model"] = self.model.state_dict()  # this is the raw key containing _orig and _module flags
        save_state["scheduler"] = self.scheduler.state_dict()
        save_state["scaler"] = self.scaler.state_dict()
        save_state["metadata"] = metadata
        torch.save(save_state, file)

    def load_training_checkpoint(self, identifier="intermediate.pth", directory=""):
        self.optimizer.zero_grad()
        file = os.path.join(directory, str(identifier))

        save_state = torch.load(file, map_location=torch.device("cpu"))
        self.model.load_state_dict(save_state["model"])  # why does this end up on GPU?
        self.optimizer.load_state_dict(save_state["optim"])
        self.scheduler.load_state_dict(save_state["scheduler"])
        self.scaler.load_state_dict(save_state["scaler"])
        log.info(f"Sucessfully loaded state with metadata {save_state['metadata']}")
        return save_state["metadata"]

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
            # Save model.safetensors, model_config.json
            save_file(self.retrieve_model_state_dict(), os.path.join(full_path, "model.safetensors"))
            # legacy save: torch.save(self.retrieve_model_state_dict(), os.path.join(full_path, "model.pth"))
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

            # Push model with safetensors:
            # This is a manual modification of model.push_to_hub which doesn't support safetensors yet
            repo_id = cfg.impl.hf_directoy_name
            if os.path.isdir(repo_id):
                working_dir = repo_id
                repo_id = repo_id.split(os.path.sep)[-1]
            else:
                working_dir = repo_id.split("/")[-1]
            repo_id = self.model._create_repo(repo_id)
            use_temp_dir = not os.path.isdir(working_dir)
            with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
                files_timestamps = self.model._get_files_timestamps(work_dir)
                # Save all files.
                self.model.save_pretrained(
                    work_dir,
                    max_shard_size="10GB",
                    safe_serialization=True,
                    state_dict=self.retrieve_model_state_dict(),
                )
                self.model._upload_modified_files(
                    work_dir,
                    repo_id,
                    files_timestamps,
                    commit_message=None,
                    token=None,
                    create_pr=False,
                )
            # Push tokenizer:
            tokenizer.push_to_hub(cfg.impl.hf_directoy_name)
            # Push config files:
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


class TorchEngineFull(TorchEngineMinimal):
    """This class mirrors deepspeed functionality. Not all changes are implemented in this version.

    See TorchEngineFull for more modifications.
    """

    def __init__(self, model, cfg_train, cfg_impl, setup=_default_setup, seq_length=128):
        """Load Engine. The model will be compiled by default."""
        super().__init__(model, cfg_train, cfg_impl, setup, seq_length)

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
        self.initial_time = time.time()

    def optimizer_step(self):
        """Requires a scheduler that is based on iterations instead of epochs."""
        super().optimizer_step()
        if self.accumulated_samples >= self.current_batch_size:
            self.schedule_curriculum()
            self.moving_average_computation()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids", "labels"]):
        """Move batch of data into device memory."""
        device_batch = super().to_device(batch)
        self.set_sequence_curriculum_(device_batch)
        return device_batch

    def set_sequence_curriculum_(self, device_batch):
        """Assume huggingface data is B S"""
        if self.sequence_curriculum:
            for key, tensor in device_batch.items():
                if self.sequence_unfold:
                    device_batch[key] = tensor.view(-1, self.current_seq_length)
                else:
                    device_batch[key] = tensor[:, : self.current_seq_length].clone()

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


def _load_optimizer(model, cfg_train, cfg_impl, initial_time):

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

    scheduler = get_schedule_fn(initial_time, cfg_train)(optimizer_to_schedule)

    return optimizer, scheduler
