"""Rewrite a simplified BERT version based on the huggingface BERT but allow for scripting to all kinds of variations."""
import torch

# import torchdynamo  # need to disable dynamo in dynamic parts

from typing import Optional

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    AttentionComponent,
    FFNComponent,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
)


def construct_fixed_cramlm(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes
    if downstream_classes is None:
        model = ScriptableLMForPreTraining(ScriptableLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(ScriptableLM(cfg_arch), cfg_arch)
    return model


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states


class ScriptableLM(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    def __init__(self, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.embedding = EmbeddingComponent(cfg_arch.embedding, cfg_arch.norm, cfg_arch.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, cfg_arch) for idx in range(cfg_arch.num_transformer_layers)])

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        hidden_states = self.embedding(input_ids)
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)

        hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)


class ScriptableLMForPreTraining(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    def __init__(self, encoder, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.encoder = encoder
        self.prediction_head = PredictionHeadComponent(cfg_arch)

        self.decoder = torch.nn.Linear(cfg_arch.embedding.embedding_dim, cfg_arch.embedding.vocab_size, bias=cfg_arch.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
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

        return {"loss": masked_lm_loss}

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


def _init_module(name, module, init_method, init_std=0.02, hidden_size=768, num_layers=12):
    std = init_std
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
