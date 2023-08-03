"""(Hopefully) seamless integration of deepspeed.

I have not used this in a while, handle the deepspeed backend with care.

"""
import torch
import os
import json
from functools import partial

import logging
from omegaconf import OmegaConf
from .utils import group_parameters, prepare_pretraining_dataloader
from .optimizers import get_schedule_fn

log = logging.getLogger(__name__)
_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


"""Todo:
* integrate batch size ramping via
https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.runtime.pipe.engine.PipelineEngine.set_train_batch_size
"""


def initialize_deepspeed(model, dataset, tokenizer, cfg_train, cfg_impl, setup=_default_setup):
    """Initialize deepspeed. Module is imported lazily here."""
    import deepspeed

    if cfg_impl.jit == "trace":
        # This variant is very experimental...
        input_setup = dict(dtype=torch.long, device=setup["device"])
        templates = torch.randint(0, model.vocab_size, (*cfg_impl.trace_shape,), **input_setup)
        labels = torch.randint(0, model.vocab_size, (*cfg_impl.trace_shape,), **input_setup)

        model.to(**setup)
        model.kwargs_forward = model.forward
        model.forward = lambda input_ids, labels: model.kwargs_forward(input_ids=input_ids, labels=labels)
        model = torch.jit.trace(model, (templates, labels), strict=False)
    elif cfg_impl.jit == "script":
        # This does not work for huggingface models
        model = torch.jit.script(model)

    model_engine, optimizer, dataloader, scheduler = deepspeed.initialize(
        config=OmegaConf.to_container(cfg_impl, resolve=True),
        model=model,
        model_parameters=group_parameters(model, cfg_train),
        lr_scheduler=get_schedule_fn(cfg_train),
        # training_data=dataset, # handle this natively
        # collate_fn=collate_fn,
    )
    # Monkey-patch checkpointing
    model_engine.save_training_checkpoint = partial(save_training_checkpoint, self=model_engine)
    model_engine.save_final_model = partial(save_final_model, model_engine)
    # And more methods
    model_engine.gradinit = partial(gradinit, self=model_engine)
    model_engine.to_device = lambda batch: to_device(self=model_engine, batch=batch, keys=["input_ids", "labels"])

    model_engine.setup = setup
    model_engine.record_batch_size = lambda: cfg_train.batch_size
    model_engine.record_tokens_per_step = lambda: tokenizer.model_max_length * cfg_impl.microbatch_size

    def step(self, batch):
        loss = self.forward(**batch)["loss"]
        self.backward(loss)
        self.optimizer_step()
        return loss.detach()

    if dataset is not None:
        dataloader = prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl)
    else:
        dataloader = None
    # dataloader = deepspeed.RepeatingLoader(dataloader)
    return model_engine, optimizer, scheduler, dataloader


def save_training_checkpoint(self, identifier, directory="checkpoints", state=None):
    """Path, identifier and additional client state. This checkpoint can be used to resume training.
    The default behavior is to save this checkpoint relative to the training working directory.
    """
    self.save_checkpoint(directory, identifier, client_state=state)


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
        self.save_checkpoint(full_path, "model")
        with open(os.path.join(full_path, "model_config.json"), "w") as file:
            json.dump(OmegaConf.to_container(cfg_arch, resolve=True), file)


def gradinit(self, dataloader, config):
    raise ValueError("GradInit not implemented for deepspeed.")


def to_device(self, batch, keys=["input_ids", "labels"]):
    """Move batch of data into device memory."""
    return {
        k: v.to(device=self.setup["device"], dtype=torch.long, non_blocking=True)
        for k, v in batch.items()
        if k in keys  # Add more keywords here if needed
    }
