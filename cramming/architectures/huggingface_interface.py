"""BERT variations based on the huggingface implementation."""

import transformers
from omegaconf import OmegaConf


def construct_huggingface_model(cfg_arch, vocab_size, downstream_classes=None):
    """construct model from given configuration. Only works if this arch exists on the hub."""
    if downstream_classes is None:
        if isinstance(cfg_arch, transformers.PretrainedConfig):
            configuration = cfg_arch
        else:
            configuration = transformers.BertConfig(**cfg_arch)
        configuration.pad_token_id = None  # Need to drop this during pretraining, otherwise leads to a graph break in a HF warning
        configuration.vocab_size = vocab_size
        model = transformers.AutoModelForMaskedLM.from_config(configuration)
        model.vocab_size = model.config.vocab_size
    else:
        if isinstance(cfg_arch, transformers.PretrainedConfig):
            configuration = cfg_arch
            configuration.num_labels = downstream_classes
            if hasattr(configuration, "arch"):
                configuration.arch["num_labels"] = downstream_classes
        else:
            configuration = OmegaConf.to_container(cfg_arch)
            configuration = transformers.BertConfig(**configuration, num_labels=downstream_classes)
        configuration.vocab_size = vocab_size

        configuration.problem_type = None  # always reset this!
        model = transformers.AutoModelForSequenceClassification.from_config(configuration)
        model.vocab_size = vocab_size
    return model
