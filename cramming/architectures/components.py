"""Basic transformer components."""

import torch

from typing import Optional, Tuple

from functools import partial
from .embeddings import SinusoidalPositional, LearnablePositional, ScaledSinosoidal
from .attention import get_attention_mechanism, FLASH
from .fused_layers import get_layer_fn

INPLACE = False


class EmbeddingComponent(torch.nn.Module):
    def __init__(self, cfg_embedding, norm, norm_eps):
        super().__init__()
        self.word_embedding = torch.nn.Embedding(
            cfg_embedding.vocab_size, cfg_embedding.embedding_dim, padding_idx=cfg_embedding.pad_token_id
        )
        if cfg_embedding.pos_embedding == "learned":
            self.pos_embedding = LearnablePositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "sinusoidal":
            self.pos_embedding = SinusoidalPositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "scaled-sinusoidal":
            self.pos_embedding = ScaledSinosoidal(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        else:
            self.pos_embedding = None

        self.dropout = torch.nn.Dropout(p=cfg_embedding.dropout_prob, inplace=INPLACE)
        if cfg_embedding.normalization:
            self.norm = _get_norm_fn(norm)(cfg_embedding.embedding_dim, eps=norm_eps)
        else:
            self.norm = torch.nn.Identity()

    def forward(self, input_ids):
        embeds = self.word_embedding(input_ids)
        if self.pos_embedding is not None:
            embeds += self.pos_embedding(input_ids)
        return self.dropout(self.norm(embeds))


class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)

        if cfg_attention.low_level_fusion:
            if hasattr(self.self_attention, "sequence_op"):
                self.self_attention.sequence_op = torch.jit.script(self.sequence_op)
            if hasattr(self.self_attention, "rotary_emb"):
                self.self_attention.rotary_emb = torch.jit.script(self.rotary_emb)
        if cfg_attention.high_level_fusion:
            self.self_attention = torch.jit.script(self.self_attention)

        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(hidden_states, attention_mask))


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    Better do this manually and choose a sensible intermed_size that is nicely divisible.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU) or getattr(self.nonlin, "original_name", "") == "GLU":
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=INPLACE)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        nonlin_fn = _get_nonlin_fn(cfg_arch.nonlin)
        if (idx + 1) % cfg_arch.ffn_layer_frequency == 0:
            self.ffn = FFNComponent(
                cfg_arch.hidden_size,
                cfg_arch.intermed_size,
                nonlin_fn,
                cfg_arch.use_bias,
            )
        else:
            self.ffn = torch.nn.Identity()

        self.norm_scheme = cfg_arch.norm_scheme
        if self.norm_scheme == "sandwich":
            self.norm3 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.norm4 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.fn = get_layer_fn(
                type="sandwich", scripting=cfg_arch.layer_fusion, dn=cfg_arch.deepnorm_scaling, drop=cfg_arch.layer_drop_theta
            )
        else:
            self.fn_training, self.fn_eval = get_layer_fn(
                self.norm_scheme,
                prob=cfg_arch.hidden_dropout_prob,
                scripting=cfg_arch.layer_fusion,
                dn=cfg_arch.deepnorm_scaling,
                drop=cfg_arch.layer_drop_theta,
            )

        if cfg_arch.deepnorm_scaling:
            self.register_buffer("alpha", torch.tensor(2 * cfg_arch.num_transformer_layers).pow(0.25))
        else:
            self.register_buffer("alpha", torch.tensor(1.0))

        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None, res_scale=1):
        if self.norm_scheme == "pre":  # On Layer Normalization in the Transformer Architecture
            if self.training:
                states = self.fn_training(states, self.attn(self.norm1(states), attention_mask), self.alpha, res_scale)
                states = self.fn_training(states, self.ffn(self.norm2(states)), self.alpha, res_scale)
            else:
                states = self.fn_eval(states, self.attn(self.norm1(states), attention_mask), self.alpha, res_scale)
                states = self.fn_eval(states, self.ffn(self.norm2(states)), self.alpha, res_scale)

        elif self.norm_scheme == "sandwich":
            states = self.fn(states, self.norm3(self.dropout(self.attn(self.norm1(states), attention_mask))), self.alpha, res_scale)
            states = self.fn(states, self.norm4(self.dropout(self.ffn(self.norm2(states)))), self.alpha, res_scale)
        else:
            if self.training:
                states = self.norm1(self.fn_training(states, self.attn(states, attention_mask), self.alpha, res_scale))
                states = self.norm2(self.fn_training(states, self.ffn(states), self.alpha, res_scale))
            else:
                states = self.norm1(self.fn_eval(states, self.attn(states, attention_mask), self.alpha, res_scale))
                states = self.norm2(self.fn_eval(states, self.ffn(states), self.alpha, res_scale))

        # this thing is simply a terrible version of
        # states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # hopefully torch 2.0 can make it obsolete

        return states


class FLASHLayer(torch.nn.Module):
    """A FLASH-quad layer."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = FLASH(cfg_arch.hidden_size, cfg_arch.attention)

        self.norm_scheme = cfg_arch.norm_scheme
        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None, res_scale=1):
        if self.norm_scheme == "pre":  # On Layer Normalization in the Transformer Architecture
            states = states + self.attn(self.norm(states), attention_mask)
        else:
            states = self.norm(states + self.attn(states, attention_mask))
        return states


class PoolingComponent(torch.nn.Module):
    def __init__(self, cfg_head, main_model_hidden_size):
        super().__init__()
        self.dense = torch.nn.Linear(main_model_hidden_size, cfg_head.head_dim) if cfg_head.include_ff_layer else torch.nn.Identity()
        self.activation = _get_nonlin_fn(cfg_head.nonlin, use_gating=False)()
        self.dropout = torch.nn.Dropout(cfg_head.classifier_dropout)
        self.pool_scheme: str = cfg_head.pooler

    def forward(self, hidden_states):
        """A variety of pooling options. Some ignore the cls token. Input needs to be B S H."""
        if self.pool_scheme == "zero_index":
            first_token_tensor = hidden_states[:, 0]
        elif self.pool_scheme == "avg":
            first_token_tensor = hidden_states.mean(dim=1)
        elif self.pool_scheme == "max":
            first_token_tensor = hidden_states.max(dim=1)[0]
        elif self.pool_scheme == "lse":
            first_token_tensor = hidden_states.logsumexp(dim=1)
        else:
            raise ValueError(f"Invalid pooling scheme {self.pool_scheme} given.")

        pooled_output = self.activation(self.dense(first_token_tensor))
        return self.dropout(pooled_output)


class PredictionHeadComponent(torch.nn.Module):
    def __init__(self, cfg_arch):
        super().__init__()

        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            output_size = cfg_arch.hidden_size
        else:
            output_size = cfg_arch.embedding.embedding_dim

        self.dense = torch.nn.Linear(cfg_arch.hidden_size, output_size, bias=cfg_arch.use_bias)
        self.nonlin = _get_nonlin_fn(cfg_arch.nonlin, use_gating=False)()
        self.norm = _get_norm_fn(cfg_arch.norm)(output_size, eps=cfg_arch.norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.norm(self.nonlin(self.dense(hidden_states)))
        return hidden_states


def _get_layer_fn(layer_macro_type):
    if layer_macro_type == "transformer":
        return TransformerLayer
    elif layer_macro_type == "FLASH":
        return FLASHLayer
    else:
        raise ValueError(f"Invalid layer type {layer_macro_type} given.")


def _get_norm_fn(norm_name):
    if norm_name == "ScaleNorm":
        norm_fn = ScaleNorm
    elif norm_name == "RMSNorm":
        norm_fn = RMSNorm
    elif norm_name == "ApexLayerNorm":
        from apex.normalization import FusedLayerNorm

        norm_fn = FusedLayerNorm
    else:
        norm_fn = getattr(torch.nn, norm_name)
    return norm_fn


def _get_nonlin_fn(nonlin_name, use_gating=True):
    if "glu" in nonlin_name.lower():
        nonlin_name = nonlin_name.split("glu")[0]
        wrap_in_glu = use_gating
    else:
        wrap_in_glu = False
    nonlin_fn = getattr(torch.nn, nonlin_name)  # dont mess this up :<
    try:
        nonlin_fn = partial(nonlin_fn, inplace=INPLACE)
        nonlin_fn()
    except TypeError:
        nonlin_fn = getattr(torch.nn, nonlin_name)

    if wrap_in_glu:
        return partial(GLU, nonlin_fn)
    else:
        return nonlin_fn


class GLU(torch.nn.Module):
    """*-GLU activation functions.

    Implementation mostly following megatron
    """

    def __init__(self, sub_activation):
        super().__init__()
        self.sub_activation = sub_activation()

    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        return self.sub_activation(gate) * x


class ScaleNorm(torch.nn.Module):
    """Quick and simple scale norm implementation.

    Do we also need FixNorm (cosine in the last layer)? It's a maybe here:
    https://github.com/lucidrains/performer-pytorch/issues/55#issuecomment-762544686
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.learnable_scale = torch.nn.Parameter(torch.tensor(float(hidden_size) ** -0.5))

    def forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=self.eps)


class RMSNorm(torch.nn.Module):
    """The RMS variant of scaling norms."""

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.learnable_scale = torch.nn.Parameter(torch.ones(hidden_size) ** -0.5)

    def forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=self.eps)


class Sequential(torch.nn.Module):
    """Modified sequential class."""

    def __init__(self, list_of_modules):
        super().__init__()
        self.seq_modules = torch.nn.ModuleList(list_of_modules)
        self.LAYOUT = self.seq_modules[0].LAYOUT

    def forward(self, states, *args, **kwargs):
        for module in self.seq_modules:
            states = module(states, *args, **kwargs)
        return states


def get_extended_attention_mask(attention_mask: torch.Tensor, input_shape: Tuple[int], causal_attention: bool = False) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.

    Method stolen from huggingface :)
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if causal_attention:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=attention_mask.device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones((batch_size, seq_length, prefix_seq_len), device=attention_mask.device, dtype=causal_mask.dtype),
                        causal_mask,
                    ],
                    axis=-1,
                )
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    # extended_attention_mask = extended_attention_mask.to(dtype=self.setup["dtype"])  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
