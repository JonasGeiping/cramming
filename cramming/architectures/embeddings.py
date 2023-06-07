"""Standard and Non-standard embedding implementations. Several implementation variations of rotary embeddings."""
import torch
import math

from typing import Tuple
from einops import repeat

# module partially stolen from pytorch examples:
class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.pe[:, : input_ids.shape[1], :]


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0 / embedding_dim**0.5]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.scale_factor * self.pe[:, : input_ids.shape[1], :]


class LearnablePositional(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embedding_dim, max_seq_length=1024):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        position_ids = self.position_ids[:, : input_ids.shape[1]]
        return self.embedding(position_ids)


# Code stolen from GPT-X:
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

        # Force fusions on batched version
        def rotate_half(x: torch.Tensor):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster
            return torch.cat((-x2, x1), dim=-1)

        def rope_fn(cos: torch.Tensor, sin: torch.Tensor, query_layer: torch.Tensor, key_layer: torch.Tensor):
            QK = torch.cat([query_layer, key_layer], dim=1)
            rotated = QK * cos[: QK.shape[0]] + rotate_half(QK) * sin[: QK.shape[0]]
            return torch.split(QK, query_layer.shape[1], dim=1)

        self.rope_fn = rope_fn  # handle fusion on module level

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        return self.rope_fn(self.cos_cached, self.sin_cached, query_layer, key_layer)

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster


class RotarySanityCheck(torch.nn.Module):
    """not again..."""

    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        # cos, sin = self.get_cos_sin_cache(key_layer)
        # cos, sin = (cos[offset : query_layer.shape[0] + offset, ...], sin[offset : query_layer.shape[0] + offset, ...])
        cos, sin = self.cos_cached, self.sin_cached
        return (query_layer * cos) + (self.rotate_half(query_layer) * sin), (key_layer * cos) + (self.rotate_half(key_layer) * sin)

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin


# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py who adapted from
# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
class RotaryEleutherAI(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    """

    _seq_len_cached: int
    # _cos_cached: Optional[torch.Tensor]
    # _sin_cached: Optional[torch.Tensor]

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        _cos_cached, _sin_cached = self._update_cos_sin_tables(torch.randn(1, 128, 1), seq_dimension=-2)
        self.register_buffer("_cos_cached", _cos_cached, persistent=False)
        self.register_buffer("_sin_cached", _sin_cached, persistent=False)

    @torch.jit.ignore
    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        # if seq_len != self._seq_len_cached:  # or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
        self._seq_len_cached = seq_len
        t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
        # Don't do einsum, it converts fp32 to fp16
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos_cached = repeat(torch.cos(freqs).to(x.dtype), "... d -> ... (d 2)")
        sin_cached = repeat(torch.sin(freqs).to(x.dtype), "... d -> ... (d 2)")

        return cos_cached, sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        # self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=seq_dimension)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension),
        )

    @staticmethod
    def rotate_half(x: torch.Tensor):
        x = x.unflatten(dim=-1, sizes=(-1, 2))
        x1, x2 = x.unbind(dim=-1)
        rotated_x = torch.stack((-x2, x1), dim=-1)
        return rotated_x.flatten(start_dim=-2)

    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dimension: int = -2):
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[: x.shape[seq_dimension], :]
        sin = sin[: x.shape[seq_dimension], :]
        if seq_dimension == -3:
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        return (x * cos) + (rotate_half(x) * sin)


class RotaryLLAMA(torch.nn.Module):
    """Facebook implementation of rotary embeddings."""

    def __init__(self, hidden_per_head, base=10000, max_seq_length=512, seq_dim: int = 0):
        super().__init__()
        self.seq_dim: int = seq_dim
        freqs_cis = self.precompute_freqs_cis(dim=hidden_per_head, end=max_seq_length * 2, theta=base)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        return self.apply_rotary_emb(query_layer, key_layer, freqs_cis=self.freqs_cis)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        freqs_cis = freqs_cis[: x.shape[self.seq_dim]]
        # shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # shape = [1, seq_length, 1, hidden_per_head]
        shape = [s if i == self.seq_dim or i == x.ndim - 1 else 1 for i, s in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
