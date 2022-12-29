"""Initialize cramming"""

from cramming.architectures import construct_model
from cramming.backend import load_backend
from cramming.data import load_pretraining_corpus, prepare_task_dataloaders
from cramming import utils

__all__ = [
    "construct_model",
    "load_backend",
    "load_pretraining_corpus",
    "prepare_task_dataloaders",
    "utils",
]


import hydra

"""Construct interfaces to some cfg folders for use in packaged installations:"""


def get_config(overrides=[]):
    """Return default hydra config."""
    with hydra.initialize(config_path="config"):
        cfg = hydra.compose(config_name="cfg", overrides=overrides)
        print(f"Loading default config {cfg.name}.")
    return cfg


def get_model_config(arch="hf-bert-tiny", overrides=[]):
    """Return default hydra config for a given attack."""
    with hydra.initialize(config_path="config/arch"):
        cfg = hydra.compose(config_name=arch, overrides=overrides)
        print(f"Loading model configuration {cfg.architecture}.")
    return cfg


def get_backend_config(backend="torch-default", overrides=[]):
    """Return default hydra config for a given attack."""
    with hydra.initialize(config_path="config/impl"):
        cfg = hydra.compose(config_name=backend, overrides=overrides)
        print(f"Loading backend {cfg.name}.")
    return cfg
