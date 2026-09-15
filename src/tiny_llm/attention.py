import mlx.core as mx
from .basics import softmax, linear

# L is seq_len, in PyTorch API it's S (source len)
# D is head_dim

# key: N.. x L x D
# value: N.. x L x D
# query: N.. x L x D
# output: N.. x L x D
# scale = 1/sqrt(D) if not specified
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    s = query @ key.swapaxes(-1, -2)
    if scale is None:
        scale = 1.0 / mx.sqrt(key.shape[-1])
    s = s * scale
    if mask is not None:
        s = s + mask
    a = softmax(s, axis=-1)
    return a @ value

# E is hidden_size or embed_dim or dims or model_dim
# H is num_heads
# D is head_dim
# L is seq_len, in PyTorch API it's S (source len)

# w_q/w_k/w_v: (H x D) x E
# output/input: N x L x E
# w_o: E x (H x D)
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
    # query, key, and value have shape N x L x E
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        # reshape to N x L x H x D
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim)
        # transpose to N x H x L x D
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # scaled dot product attention
        output = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        # transpose back to N x L x H x D
        output = output.transpose(0, 2, 1, 3)
        # reshape back to N x L x E
        output = output.reshape(output.shape[0], output.shape[1], self.hidden_size)
        # linear projection
        output = linear(output, self.wo)
        return output
        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    return mx.triu(
            mx.full((L, S), -mx.inf),
            k= S - L + 1,
        ).astype(dtype)

# N.. is zero or more dimensions for batches
# H_q is the number of query heads
# H is the number of key/value heads (H_q must be divisible by H)
# L is the query sequence length
# S is the key/value sequence length
# D is the head dimension

# query: N.. x H_q x L x D
# key: N.. x H x S x D
# value: N.. x H x S x D
# mask: N.. x H_q x L x S
# output: N.. x H_q x L x D
def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    h_q = query.shape[-3]
    h = key.shape[-3]
    group_size = h_q // h

    batch_shape = query.shape[:-3]
    l = query.shape[-2]
    d = query.shape[-1]
    s = key.shape[-2]

    # [..., H_q, L, D] -> [..., H, G, L, D]
    query = query.reshape(
        *batch_shape,
        h,
        group_size,
        l,
        d,
    )

    # [..., H, S, D] -> [..., H, 1, S, D]
    key = mx.expand_dims(key, axis=-3)
    value = mx.expand_dims(value, axis=-3)

    if mask is not None:
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unknown mask type: {mask!r}")

            # [L, S]，自动共享给所有 batch/head/group
            mask = causal_mask(l, s, query.dtype)
        else:
            # 用户传入的是 [..., H_q, L, S]
            # 此时才需要 H_q -> H × G
            mask = mask.reshape(
                *mask.shape[:-3],
                h,
                group_size,
                l,
                s,
            )

    output = scaled_dot_product_attention_simple(
        query,
        key,
        value,
        scale=scale,
        mask=mask,
    )

    # [..., H, G, L, D] -> [..., H_q, L, D]
    return output.reshape(
        *batch_shape,
        h_q,
        l,
        d,
    )



def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
