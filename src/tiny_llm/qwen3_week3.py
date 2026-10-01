from typing import Any

import mlx.core as mx

from .attention import paged_attention, scaled_dot_product_attention_grouped
from .embedding import QuantizedEmbedding
from .kv_cache import TinyKvCache
from .moe import Moe
from .paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from .quantize import QuantizedWeights, quantized_linear
from .week2_kernels import FastRMSNorm, FastRoPE, swiglu


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_paged_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = FastRoPE(head_dim, max_seq_len, theta)
        self.q_norm = FastRMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = FastRMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.use_paged_attention = use_paged_attention

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = quantized_linear(x, self.wq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        k = quantized_linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        v = quantized_linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, offset=offsets)
        k = self.rope(k, offset=offsets)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_paged_attention:
            metadata = cache.update_and_fetch_paged(
                k,
                v,
                mask_length=L,
                mask=mask,
            )
            out = paged_attention(
                q,
                metadata.key_pages,
                metadata.value_pages,
                metadata.block_table,
                metadata.context_lens,
                metadata.page_size,
                scale=self.scale,
                mask=metadata.mask,
            )
        else:
            k, v, _, mask = cache.update_and_fetch(
                k,
                v,
                mask_length=L,
                mask=mask,
            )
            out = scaled_dot_product_attention_grouped(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=self.scale,
                mask=mask,
            ).astype(x.dtype)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return quantized_linear(out, self.wo)


class Qwen3MLP:
    def __init__(self, dim, hidden_dim, w_gate, w_up, w_down):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return quantized_linear(
            swiglu(
                quantized_linear(x, self.w_gate),
                quantized_linear(x, self.w_up),
            ),
            self.w_down,
        )


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        mlp: Qwen3MLP | Moe,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_paged_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.input_layernorm = FastRMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_paged_attention=use_paged_attention,
        )

    def __call__(self, x, offset, cache, mask=None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def is_qwen3_moe_sparse_layer(args: Any, layer_idx: int) -> bool:
    return (
        getattr(args, "num_experts", 0) > 0
        and layer_idx not in getattr(args, "mlp_only_layers", [])
        and (layer_idx + 1) % getattr(args, "decoder_sparse_step", 1) == 0
    )


class Qwen3ModelWeek3:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
        enable_paged_attention: bool = True,
        use_mlx_quantized_linear: bool = True,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.page_size = page_size
        self.page_pools = [
            TinyKvPagedPool(page_size=page_size)
            for _ in range(self.num_hidden_layers)
        ]
        self.precision = mx.bfloat16
        self.use_mlx_quantized_linear = use_mlx_quantized_linear and not getattr(
            mlx_model.args, "num_experts", 0
        )

        def weights(layer):
            return QuantizedWeights.from_mlx_layer(
                layer,
                use_simdgroup_matmul=True,
                use_split_k_matmul=True,
                use_mlx_quantized_linear=self.use_mlx_quantized_linear,
            )

        self.embedding = QuantizedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=weights(mlx_model.model.embed_tokens),
            use_custom_kernel=True,
        )
        self.layers_inner = []
        for i in range(self.num_hidden_layers):
            mlx_layer = mlx_model.model.layers[i]
            if is_qwen3_moe_sparse_layer(mlx_model.args, i):
                mlp = Moe(
                    w_router=weights(mlx_layer.mlp.gate),
                    w_gate=weights(mlx_layer.mlp.switch_mlp.gate_proj),
                    w_up=weights(mlx_layer.mlp.switch_mlp.up_proj),
                    w_down=weights(mlx_layer.mlp.switch_mlp.down_proj),
                    num_experts_per_tok=mlx_model.args.num_experts_per_tok,
                    norm_topk_prob=mlx_model.args.norm_topk_prob,
                )
            else:
                mlp = Qwen3MLP(
                    mlx_model.args.hidden_size,
                    mlx_model.args.intermediate_size,
                    weights(mlx_layer.mlp.gate_proj),
                    weights(mlx_layer.mlp.up_proj),
                    weights(mlx_layer.mlp.down_proj),
                )
            self.layers_inner.append(
                Qwen3TransformerBlock(
                    num_attention_heads=mlx_model.args.num_attention_heads,
                    num_kv_heads=mlx_model.args.num_key_value_heads,
                    hidden_size=mlx_model.args.hidden_size,
                    head_dim=mlx_model.args.head_dim,
                    rms_norm_eps=mlx_model.args.rms_norm_eps,
                    wq=weights(mlx_layer.self_attn.q_proj),
                    wk=weights(mlx_layer.self_attn.k_proj),
                    wv=weights(mlx_layer.self_attn.v_proj),
                    wo=weights(mlx_layer.self_attn.o_proj),
                    q_norm=mlx_layer.self_attn.q_norm.weight,
                    k_norm=mlx_layer.self_attn.k_norm.weight,
                    w_input_layernorm=mlx_layer.input_layernorm.weight,
                    w_post_attention_layernorm=mlx_layer.post_attention_layernorm.weight,
                    mlp=mlp,
                    max_seq_len=mlx_model.args.max_position_embeddings,
                    theta=mlx_model.args.rope_theta,
                    use_paged_attention=enable_paged_attention,
                )
            )
        self.norm = FastRMSNorm(
            self.hidden_size,
            mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        self.w_lm_head = (
            None
            if mlx_model.args.tie_word_embeddings
            else weights(mlx_model.lm_head)
        )

    def create_kv_cache(self) -> list[TinyKvCache]:
        return [TinyKvPagedCache(pool=pool) for pool in self.page_pools]

    def __call__(self, inputs, offset, cache, logits_to_keep=None) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        return self.embedding.as_linear(h)
