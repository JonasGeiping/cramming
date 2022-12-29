"""Rewrite a simplified BERT version based on the huggingface BERT but allow for scripting to all kinds of variations."""
import torch

# import torchdynamo  # need to disable dynamo in dynamic parts

from typing import Optional

from .components import (
    _get_layer_fn,
    _get_norm_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    Sequential,
    get_extended_attention_mask,
)


def construct_scriptable_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes
    if downstream_classes is None:
        model = ScriptableLMForPreTraining(ScriptableLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(ScriptableLM(cfg_arch), cfg_arch)
    return model


class ScriptableLM(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    def __init__(self, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.embedding = EmbeddingComponent(cfg_arch.embedding, cfg_arch.norm, cfg_arch.norm_eps)
        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            self.input_projection = torch.nn.Identity()
        else:
            self.input_projection = torch.nn.Linear(
                cfg_arch.embedding.embedding_dim,
                cfg_arch.hidden_size,
                bias=cfg_arch.use_bias,
            )

        layer_fn = _get_layer_fn(cfg_arch.layer_macro_type)
        if cfg_arch.recurrent_layers is None:
            self.layers = torch.nn.ModuleList([layer_fn(idx, cfg_arch) for idx in range(cfg_arch.num_transformer_layers)])
        else:
            core_block = Sequential([layer_fn(idx, cfg_arch) for idx in range(cfg_arch.recurrent_layers)])
            self.layers = torch.nn.ModuleList([core_block for _ in range(cfg_arch.num_transformer_layers)])

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.gradient_checkpointing = cfg_arch.gradient_checkpointing
        self.layer_drop_theta = cfg_arch.layer_drop_theta
        self.register_buffer("p", torch.tensor(1.0))  # Layer scaling factor # Assign this only once

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)
        hidden_states = self.input_projection(self.embedding(input_ids))

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Main transformer blocks:
        if self.gradient_checkpointing and self.training:
            # Hide this away from any jit-ing...
            hidden_states = self.forward_checkpointed(hidden_states, attention_mask)
        else:
            if self.layer_drop_theta is None:
                for i, layer_module in enumerate(self.layers):
                    hidden_states = layer_module(hidden_states, attention_mask, self.p)
            else:
                p = self.p.clone()
                step = (1 - self.layer_drop_theta) / len(self.layers)
                for i, layer_module in enumerate(self.layers):
                    p = p - step
                    if torch.bernoulli(p):
                        hidden_states = layer_module(hidden_states, attention_mask, res_scale=1 / p)
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)

    @torch.jit.ignore
    def forward_checkpointed(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        if self.layer_drop_theta is None:
            for i, layer_module in enumerate(self.layers):
                hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)
        else:
            p = self.p.clone()
            step = (1 - self.layer_drop_theta) / len(self.layers)
            for i, layer_module in enumerate(self.layers):
                p = p - step
                if torch.bernoulli(p):
                    hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask, res_scale=1 / p)
        return hidden_states


class ScriptableLMForPreTraining(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    def __init__(self, encoder, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.encoder = encoder
        if not cfg_arch.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(cfg_arch)
        else:
            self.prediction_head = torch.nn.Linear(
                cfg_arch.hidden_size,
                cfg_arch.embedding.embedding_dim,
                bias=cfg_arch.use_bias,
            )

        if cfg_arch.loss == "szegedy":
            self.decoder = torch.nn.Identity()
        else:
            if self.cfg.tie_weights:
                self.decoder = torch.nn.Linear(cfg_arch.embedding.embedding_dim, cfg_arch.embedding.vocab_size, bias=cfg_arch.decoder_bias)
                self.decoder.weight = self.encoder.embedding.word_embedding.weight
            else:
                self.decoder = torch.nn.Linear(cfg_arch.hidden_size, cfg_arch.embedding.vocab_size, bias=cfg_arch.decoder_bias)

        self.loss_fn = _get_loss_fn(cfg_arch.loss, z_loss_factor=cfg_arch.z_loss_factor, embedding=self.encoder.embedding.word_embedding)
        self.sparse_prediction = self.cfg.sparse_prediction
        self.vocab_size = cfg_arch.embedding.vocab_size

        for name, module in self.named_modules():
            _init_module(
                name,
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction:
            masked_lm_loss = self._forward_dynamic(outputs, labels)
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return dict(loss=masked_lm_loss)

    # Sparse prediction can have an unpredictable number of entries in each batch
    # depending on how MLM is running
    # for this reason, the code has to fall back to eager mode there
    # @torchdynamo.disable
    def _forward_dynamic(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if labels is not None:
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            outputs = outputs[mask_positions]
            labels = labels[mask_positions]

        outputs = self.decoder(self.prediction_head(outputs))
        if labels is not None:
            masked_lm_loss = self.loss_fn(outputs, labels)
        else:
            masked_lm_loss = outputs.new_zeros((1,))
        return masked_lm_loss


class ScriptableLMForSequenceClassification(torch.nn.Module):
    """Classification head and pooler."""

    def __init__(self, encoder, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.encoder = encoder
        self.pooler = PoolingComponent(cfg_arch.classification_head, cfg_arch.hidden_size)
        self.head = torch.nn.Linear(cfg_arch.classification_head.head_dim, cfg_arch.num_labels)

        self.problem_type = None
        self.num_labels = self.cfg.num_labels

        for name, module in self.named_modules():
            _init_module(
                name,
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.pooler(self.encoder(input_ids, attention_mask)))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.cfg.num_labels == 1:
                    self.problem_type = "regression"
                elif self.cfg.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


def _get_loss_fn(loss_fn_name, z_loss_factor=0.0, embedding=torch.nn.Identity()):
    if loss_fn_name == "cross-entropy":
        if z_loss_factor > 0:
            from .losses import CrossEntropyWithZLoss

            return torch.jit.script(CrossEntropyWithZLoss(z_loss_factor=z_loss_factor))
        else:
            return torch.nn.CrossEntropyLoss()
    # elif loss_fn_name == "adaptive-cross-entropy":
    #     loss_fn = torch.nn.AdaptiveLogSoftmaxWithLoss(
    #         in_features,
    #         n_classes,
    #         cutoffs,
    #         div_value=4.0,
    #         head_bias=False,
    #     )
    elif loss_fn_name == "MSE":
        assert z_loss_factor == 0
        from .losses import MSELoss

        return torch.jit.script(MSELoss())
    elif loss_fn_name == "MSEf":
        assert z_loss_factor == 0
        from .losses import MSELossFast

        return torch.jit.script(MSELossFast())
    elif loss_fn_name == "L1":
        assert z_loss_factor == 0
        from .losses import L1Loss

        return torch.jit.script(L1Loss())

    elif loss_fn_name == "FocalLoss":
        assert z_loss_factor == 0
        from .losses import FocalLoss

        return torch.jit.script(FocalLoss())

    elif loss_fn_name == "IncorrectLoss":
        assert z_loss_factor == 0
        from .losses import IncorrectCrossEntropyLoss

        return torch.jit.script(IncorrectCrossEntropyLoss())

    elif loss_fn_name == "szegedy":
        from .losses import SzegedyLoss

        return torch.jit.script(SzegedyLoss(embedding))
    else:
        raise ValueError(f"Invalid loss fn {loss_fn_name} given.")


def _init_module(name, module, init_method, init_std=0.02, hidden_size=768, num_layers=12):
    if init_method == "normal":
        std = init_std
    elif init_method == "small":
        # Transformers without Tears: Improving
        # the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010)
        std = torch.as_tensor(2 / (5 * hidden_size)).sqrt()
    elif init_method == "megatron":
        std = torch.as_tensor(1 / (3 * hidden_size)).sqrt()
    elif init_method == "wang":
        std = 2 / num_layers / torch.as_tensor(hidden_size).sqrt()
    elif init_method == "deepnorm":
        std = torch.as_tensor(8 * num_layers).pow(-0.25)  # todo: apply this only to some layers
    else:
        raise ValueError(f"Invalid init method {init_method} given.")

    if isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
