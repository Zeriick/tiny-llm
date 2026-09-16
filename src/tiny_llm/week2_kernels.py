import mlx.core as mx
from extensions import tiny_llm_ext


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # The C++ primitive requires the input and learned weight to have the
        # same floating-point dtype and expects contiguous buffers for its
        # backend implementation.
        return tiny_llm_ext.rms_norm(
            mx.contiguous(x),
            mx.contiguous(self.weight.astype(x.dtype)),
            self.eps,
        )


def normalize_rope_offsets(
    offset: int | list[int] | mx.array, batch_size: int
) -> mx.array:
    """Build one int32 offset per batch row; reuse already normalized arrays."""
    if isinstance(offset, int):
        return mx.full((batch_size,), offset, dtype=mx.int32)
    if isinstance(offset, list):
        if len(offset) != batch_size:
            raise ValueError("FastRoPE needs one offset per batch row")
        offset = mx.array(offset, dtype=mx.int32)
    if not isinstance(offset, mx.array):
        raise TypeError("FastRoPE offset must be an int, a list, or an MLX array")
    if offset.ndim == 0:
        return mx.full((batch_size,), offset, dtype=mx.int32)
    if offset.shape != (batch_size,):
        raise ValueError("FastRoPE needs one offset per batch row")
    return offset if offset.dtype == mx.int32 else offset.astype(mx.int32)


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        if x.ndim != 4:
            raise ValueError("FastRoPE expects x with shape [B,L,H,D]")
        # Model calls provide a shared int32 array, so this reuses it directly.
        offsets = normalize_rope_offsets(offset, x.shape[0])
        return tiny_llm_ext.rope(
            x,
            offsets,
            self.dims,
            self.base,
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fuse SiLU(gate) * up; the extension validates and prepares both inputs."""
    return tiny_llm_ext.swiglu(gate, up)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    return mx.triu(
            mx.full((L, S), -mx.inf),
            k= S - L + 1,
        ).astype(dtype)
    
def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    s = query @ key.swapaxes(-1, -2)
    if scale is None:
        scale = 1.0 / mx.sqrt(key.shape[-1])
    s = s * scale
    if mask is not None:
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unknown mask type: {mask!r}")

            # [L, S]，自动共享给所有 batch/head/group
            mask = causal_mask(query.shape[-2], key.shape[-2], query.dtype)
    if mask is not None:
        s = s + mask
    a = mx.softmax(s, axis=-1)
    return a @ value


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
