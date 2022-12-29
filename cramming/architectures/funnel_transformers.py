"""Variations on downsampling transformers."""

import torch

from typing import Optional
from torch.nn.functional import dropout

from .components import _get_norm_fn, _get_nonlin_fn, EmbeddingComponent, FFNComponent, get_extended_attention_mask
from .attention import FunnelAttention
from .scriptable_bert import ScriptableLMForPreTraining, ScriptableLMForSequenceClassification

INPLACE = False


def construct_scriptable_funnel(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes
    if downstream_classes is None:
        model = ScriptableLMForPreTraining(ScriptableFunnelLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(ScriptableFunnelLM(cfg_arch), cfg_arch)
    return model


class FunnelAttentionComponent(torch.nn.Module):
    def __init__(self, hidden_size: int, cfg_attention, use_bias: bool = True, length_factor: float = 1.0):
        super().__init__()
        assert cfg_attention.type == "funnel"
        self.self_attention = FunnelAttention(hidden_size, cfg_attention, length_factor)
        if cfg_attention.high_level_fusion:
            self.self_attention = torch.jit.script(self.self_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(hidden_states, attention_mask))


class FunnelLayer(torch.nn.Module):
    """A funnel layer."""

    def __init__(self, cfg_arch, seq_length_in: int, seq_length_out: int):
        super().__init__()
        self.dropout_prob: float = cfg_arch.hidden_dropout_prob
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)

        self.seq_length_in: int = seq_length_in
        self.seq_length_out: int = seq_length_out

        self.length_factor: float = seq_length_out / seq_length_in
        self.attn = FunnelAttentionComponent(cfg_arch.hidden_size, cfg_arch.attention, cfg_arch.use_bias, self.length_factor)

        nonlin_fn = _get_nonlin_fn(cfg_arch.nonlin)
        self.ffn = FFNComponent(cfg_arch.hidden_size, cfg_arch.intermed_size, nonlin_fn, cfg_arch.use_bias)

        assert cfg_arch.norm_scheme == "pre"
        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):

        if self.length_factor < 1:
            new_states = states.view(int(1 / self.length_factor), self.seq_length_out, states.shape[1], states.shape[2]).mean(dim=0)
        elif self.length_factor > 1:
            new_states = states.repeat_interleave(int(self.length_factor), dim=0, output_size=self.seq_length_out)
        else:
            new_states = states

        if attention_mask is not None:
            reduced_attention_mask = attention_mask.view(states.shape[1], 1, 1, states.shape[0], -1).max(dim=-1)[0]
        else:
            reduced_attention_mask = attention_mask

        if self.training:
            states = new_states + dropout(self.attn(self.norm1(states), reduced_attention_mask), p=self.dropout_prob, training=True)
            states = states + dropout(self.ffn(self.norm2(states)), p=self.dropout_prob, training=True)
        else:
            states = new_states + dropout(self.attn(self.norm1(states), reduced_attention_mask), p=self.dropout_prob, training=False)
            states = states + dropout(self.ffn(self.norm2(states)), p=self.dropout_prob, training=False)

        return states


class ScriptableFunnelLM(torch.nn.Module):
    """A funnel transformer variation. For now only implemented for fixed sequence lengths, but this is not a necessary limitation."""

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

        self.num_transformer_layers = len(cfg_arch.setup)
        layers = []
        seq_length_in = cfg_arch.setup[0]
        for idx, layer_spec in enumerate(cfg_arch.setup[1:]):
            seq_length_out = layer_spec
            layers.append(torch.jit.script(FunnelLayer(cfg_arch, seq_length_in, seq_length_out)))
            seq_length_in = layer_spec
        self.cutoff: int = torch.as_tensor(cfg_arch.setup).argmin().item() - 1

        self.layers = torch.nn.ModuleList(layers)

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.cfg.attention.causal_attention)
        hidden_states = self.input_projection(self.embedding(input_ids))

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Main transformer blocks:
        state_list = [hidden_states]
        for i, layer_module in enumerate(self.layers):
            # normal blocks
            hidden_states = layer_module(hidden_states, attention_mask)

            # with unet type residuals
            if i < self.cutoff:
                state_list.append(hidden_states)
            elif i > self.cutoff:
                shortcut_state = state_list.pop()
                hidden_states = hidden_states + shortcut_state

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)
