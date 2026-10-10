import mlx.core as mx
from extensions import tiny_llm_ext
from extensions.tiny_llm_ext import _ext as tiny_llm_ext_private
from .attention import scaled_dot_product_attention_grouped

_NO_ATTENTION_MASK = mx.zeros((1,), dtype=mx.float32)
DENSE_PREFILL_MIN_QUERY = 9
# Native BF16 BQ32/BK16 prefill; unsupported model shapes use readable attention.
DENSE_PREFILL_MMA_IMPLEMENTED = True


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps
        # Counts scheduled GPU paths by width, not completed GPU executions.
        self.dispatch_counts = {"register_cached": 0, "fixed_width_fallback": 0}

    def __call__(self, x: mx.array) -> mx.array:
        # The C++ primitive requires the input and learned weight to have the
        # same floating-point dtype and expects contiguous buffers for its
        # backend implementation.
        result = tiny_llm_ext.rms_norm(
            mx.contiguous(x),
            mx.contiguous(self.weight.astype(x.dtype)),
            self.eps,
        )
        if mx.default_device() == mx.gpu:
            path = "register_cached" if x.shape[-1] <= 4096 else "fixed_width_fallback"
            self.dispatch_counts[path] += 1
        return result


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
    result = scaled_dot_product_attention_grouped(
        query.astype(mx.float32), key.astype(mx.float32), value.astype(mx.float32),
        scale=scale, mask=mask,
    )
    if isinstance(mask, mx.array):
        # An all-masked row has zero probability mass, including idle batch slots.
        fully_masked = mx.all(mask == -mx.inf, axis=-1, keepdims=True)
        result = mx.where(fully_masked, mx.zeros_like(result), result)
    return result.astype(query.dtype)


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    # Retired bounded-decode experiment keeps a working compatibility name.
    return scaled_dot_product_attention(query, key, value, scale, mask)


def _prepare_dense_attention_inputs(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | str | None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, bool, bool]:
    """Validate and flatten grouped-query inputs for the Day 5 native seam."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("dense attention expects [B,H,L,D] query, key, and value")
    batch_size, num_heads, query_length, head_dim = query.shape
    key_batch_size, num_kv_heads, context_length, key_head_dim = key.shape
    if batch_size != key_batch_size or key.shape != value.shape:
        raise ValueError("query, key, and value batch dimensions must match")
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim != key_head_dim or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")

    query = mx.contiguous(query.reshape(batch_size * num_heads, query_length, head_dim))
    key = mx.contiguous(
        key.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    value = mx.contiguous(
        value.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    is_causal = isinstance(mask, str) and mask == "causal"
    has_mask = isinstance(mask, mx.array)
    if has_mask:
        mask = mx.broadcast_to(
            mask, (batch_size, num_heads, query_length, context_length)
        )
        mask = mx.contiguous(
            mask.astype(mx.float32).reshape(
                batch_size * num_heads, query_length, context_length
            )
        )
    else:
        mask = _NO_ATTENTION_MASK
    return query, key, value, mask, is_causal, has_mask


def dense_prefill_attention_mma(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Validate model-facing GQA arrays and run native BF16 tiled prefill."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("dense attention expects [B,H,L,D] query, key, and value")
    expected_shape = query.shape
    _, num_heads, query_length, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if query.dtype != mx.bfloat16 or head_dim != 128:
        raise ValueError("tiled dense prefill requires BF16 query/key/value with D=128")
    if query_length < DENSE_PREFILL_MIN_QUERY:
        raise ValueError(f"tiled dense prefill requires L >= {DENSE_PREFILL_MIN_QUERY}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("tiled dense prefill requires matching BF16 query/key/value")
    prepared = _prepare_dense_attention_inputs(query, key, value, mask)
    if not DENSE_PREFILL_MMA_IMPLEMENTED:
        return scaled_dot_product_attention(query, key, value, scale, mask)
    query, key, value, mask, is_causal, has_mask = prepared
    result = tiny_llm_ext_private._dense_attention_prefill_mma(
        query,
        key,
        value,
        mask,
        scale,
        is_causal,
        has_mask,
        num_heads,
        num_kv_heads,
    )
    return result.reshape(expected_shape)
