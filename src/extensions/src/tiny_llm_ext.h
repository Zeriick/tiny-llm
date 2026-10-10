#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(const char *path);

///////////////////////////////////////////////////////////////////////////////
// Week 2, Day 3: quantized matrix multiplication
///////////////////////////////////////////////////////////////////////////////

mx::array quantized_matmul(
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int bits,
    const mx::array& a,
    const mx::array& b,
    bool transpose_b = false,
    bool use_simdgroup = true,
    bool use_split_k = false,
    mx::StreamOrDevice s = {});

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(
        mx::Stream stream,
        int group_size,
        int bits,
        bool transpose_b,
        bool use_simdgroup,
        bool use_split_k)
        : mx::Primitive(stream),
          group_size_(group_size),
          bits_(bits),
          transpose_b_(transpose_b),
          use_simdgroup_(use_simdgroup),
          use_split_k_(use_split_k) {}

    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::vector<mx::array> jvp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override;
    std::vector<mx::array> vjp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) override;
    void print(std::ostream& os);
    const char* name() const override { return "QuantizedMatmul"; }
    bool is_equivalent(const mx::Primitive& other) const override;

private:
    int group_size_;
    int bits_;
    bool transpose_b_;
    bool use_simdgroup_;
    bool use_split_k_;
};

///////////////////////////////////////////////////////////////////////////////
// Legacy local declaration retained for the existing branch.
///////////////////////////////////////////////////////////////////////////////

mx::array flash_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& mask,
    float scale,
    bool is_causal,
    int num_kv_heads,
    int num_heads,
    mx::StreamOrDevice s = {});

class FlashAttention : public mx::Primitive {
public:
    explicit FlashAttention(
        mx::Stream stream,
        float scale,
        bool is_causal,
        int num_kv_heads,
        int num_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          num_kv_heads_(num_kv_heads),
          num_heads_(num_heads) {}

    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::vector<mx::array> jvp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override;
    std::vector<mx::array> vjp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) override;
    void print(std::ostream& os);
    const char* name() const override { return "FlashAttention"; }
    bool is_equivalent(const mx::Primitive& other) const override;

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

///////////////////////////////////////////////////////////////////////////////
// Upstream cumulative interfaces retained as learner-owned stubs.
///////////////////////////////////////////////////////////////////////////////

// Week 3, Day 4.
mx::array quantized_embedding(
    const mx::array& indices,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& weight,
    int group_size,
    int bits,
    mx::StreamOrDevice s = {});

class QuantizedEmbedding : public mx::Primitive {
public:
    explicit QuantizedEmbedding(mx::Stream stream) : mx::Primitive(stream) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("QuantizedEmbedding has no vmap implementation.");
    }
    const char* name() const override { return "QuantizedEmbedding"; }
};

// Week 2, Day 4.
mx::array rms_norm(
    const mx::array& x,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice s = {});
mx::array rope(
    const mx::array& x,
    const mx::array& offsets,
    int dims,
    float base,
    bool traditional,
    mx::StreamOrDevice s = {});
mx::array swiglu(
    const mx::array& gate,
    const mx::array& up,
    mx::StreamOrDevice s = {});

class Week2RMSNorm : public mx::Primitive {
public:
    Week2RMSNorm(mx::Stream stream, float eps) : mx::Primitive(stream), eps_(eps) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Week2RMSNorm has no vmap implementation.");
    }
    const char* name() const override { return "Week2RMSNorm"; }

private:
    float eps_;
};

class Week2RoPE : public mx::Primitive {
public:
    Week2RoPE(mx::Stream stream, int dims, float base, bool traditional)
        : mx::Primitive(stream), dims_(dims), base_(base), traditional_(traditional) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Week2RoPE has no vmap implementation.");
    }
    const char* name() const override { return "Week2RoPE"; }

private:
    int dims_;
    float base_;
    bool traditional_;
};

class Week2SwiGLU : public mx::Primitive {
public:
    explicit Week2SwiGLU(mx::Stream stream) : mx::Primitive(stream) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Week2SwiGLU has no vmap implementation.");
    }
    const char* name() const override { return "Week2SwiGLU"; }
};

// Week 2, Day 6: implement online-softmax decode attention.
mx::array decode_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& mask,
    float scale,
    bool is_causal,
    bool has_mask,
    int num_heads,
    int num_kv_heads,
    mx::StreamOrDevice s = {});

class Week2DecodeAttention : public mx::Primitive {
public:
    Week2DecodeAttention(
        mx::Stream stream,
        float scale,
        bool is_causal,
        bool has_mask,
        int num_heads,
        int num_kv_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          has_mask_(has_mask),
          num_heads_(num_heads),
          num_kv_heads_(num_kv_heads) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Week2DecodeAttention has no vmap implementation.");
    }
    const char* name() const override { return "Week2DecodeAttention"; }

private:
    float scale_;
    bool is_causal_;
    bool has_mask_;
    int num_heads_;
    int num_kv_heads_;
};

// Week 3, Day 3.
mx::array paged_cache_update(
    const mx::array& pages,
    const mx::array& values,
    int page_id,
    int start,
    mx::StreamOrDevice s = {});

class PagedCacheUpdate : public mx::Primitive {
public:
    PagedCacheUpdate(mx::Stream stream, int page_id, int start)
        : mx::Primitive(stream), page_id_(page_id), start_(start) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("PagedCacheUpdate has no vmap implementation.");
    }
    const char* name() const override { return "PagedCacheUpdate"; }

private:
    int page_id_;
    int start_;
};

// Week 3, Day 4; Day 5 keeps the same interface for the long-prefill schedule.
mx::array paged_attention(
    const mx::array& q,
    const mx::array& key_pages,
    const mx::array& value_pages,
    const mx::array& block_table,
    const mx::array& context_lens,
    float scale,
    bool is_causal,
    int num_kv_heads,
    int num_heads,
    mx::StreamOrDevice s = {});

class PagedAttention : public mx::Primitive {
public:
    PagedAttention(mx::Stream stream, float scale, bool is_causal, int num_kv_heads, int num_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          num_kv_heads_(num_kv_heads),
          num_heads_(num_heads) {}
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("PagedAttention has no vmap implementation.");
    }
    const char* name() const override { return "PagedAttention"; }

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

///////////////////////////////////////////////////////////////////////////////
// Week 2, Day 5: tiled dense prefill attention -- implementation contract
//
// Implemented by the native entry, GPU primitive, and matching Metal kernel.
// The public Python adapter owns broadcasting, contiguous copies, and reshaping.
// The native entry must also validate its inputs before constructing a primitive.
//
// 1. Native array layout (batch and head axes have already been flattened):
//      q:      [B * Hq,  L, 128], contiguous BF16
//      k, v:   [B * Hkv, S, 128], contiguous BF16, identical shapes
//      output: [B * Hq,  L, 128], contiguous BF16, same shape as q
//    Hq = num_heads > 0, Hkv = num_kv_heads > 0, Hq % Hkv == 0.
//    Require L >= 9, S > 0, and matching B inferred from q and k/v.
//    For cached causal self-attention require S >= L.
//    The model handles short queries / other dtypes or head dimensions via its
//    readable fallback; this direct primitive rejects unsupported inputs.
//
// 2. GQA indexing (never replicate K/V to Hq heads):
//      q_head_flat = b * Hq + h
//      kv_head_flat = b * Hkv + h / (Hq / Hkv)
//    Q and output use q_head_flat; K and V use kv_head_flat.
//
// 3. Score and mask semantics:
//      score(i, j) = scale * dot(q[i], k[j]) + additive_mask(i, j)
//    scale is supplied by the caller; apply it once, before the additive mask.
//    is_causal: source j is visible iff j <= S - L + i, where i is the query
//    row within the full L rows, NOT the row within a BQ tile.
//    has_mask: mask is contiguous FP32 [B * Hq, L, S], already broadcast.
//    !has_mask: mask is a placeholder array; do not read it or require its shape
//    to match scores. Both flags false means unmasked attention. If both flags
//    are true, apply both constraints (Python normally selects one mask mode).
//
// 4. GPU ownership and launch contract:
//    BQ = 32 query rows, BK = 16 source positions, D = 128.
//    One threadgroup owns one (q_head_flat, query_tile), and iterates over S.
//    dispatch_threadgroups: grid = (ceil(L / BQ), B * Hq, 1), group = (128,1,1).
//    Metal group_id.x = query_tile, group_id.y = q_head_flat.
//    Require SIMD width 32 (four SIMD groups) and kernel thread limit >= 128.
//    These are 128 cooperating threads, NOT one thread per query row.
//    Cooperatively stage Q and K/V in threadgroup memory; use SIMD-group matrix
//    operations for Q @ K^T and weighted V. Synchronize before consuming shared
//    tiles and before reusing K storage for V or the next source tile.
//
// 5. Metal buffer slots (keep eval_gpu and the kernel signature aligned):
//      0=q, 1=k, 2=v, 3=mask, 4=output
//      5=L (int32), 6=S (int32), 7=scale (float32)
//      8=is_causal (uint32), 9=has_mask (uint32)
//      10=Hq (int32), 11=Hkv (int32)
//    Convert host bools to uint32 flags when binding; do not depend on bool ABI.
//    Shared-memory layout/size must agree between host and Metal; it is separate
//    from these device-buffer slots and must include any tile padding.
//
// 6. Online softmax, independently for EACH query row:
//    Keep FP32 m=-infinity, l=0 and a[D]=0 across source tiles.
//    For a tile with at least one valid score in this row:
//      next_m = max(m, max(tile_scores))
//      alpha = (l == 0) ? 0 : exp(m - next_m)
//      p[j] = valid(j) ? exp(score[j] - next_m) : 0
//      l = alpha * l + sum(p)
//      a = alpha * a + sum_j(p[j] * v[j])
//      m = next_m
//    Rescale BOTH the numerator a and denominator l. Skip a row's state update
//    for tiles with no valid scores; avoid -infinity - (-infinity).
//    Finally output = (l > 0) ? a / l : 0, with final conversion to BF16.
//    Do not allocate a full L*S score/probability workspace. A caller-supplied
//    additive mask can still occupy L*S entries per query head.
//
// 7. Tails and execution:
//    Guard Q/K/V loads and output stores for partial BQ/BK tiles. Padded source
//    positions contribute zero probability, not score zero. Inactive query rows
//    must not write output or bypass barriers needed by the rest of the group.
//    Fully masked rows return finite zeros. eval_cpu reports a GPU-only error.
//    DENSE_PREFILL_MMA_IMPLEMENTED selects this native path in Python.
//    Validate native dispatch as well as numerical output in regression checks.
///////////////////////////////////////////////////////////////////////////////
mx::array _dense_attention_prefill_mma(
    const mx::array &q,
    const mx::array &k,
    const mx::array &v,
    const mx::array &mask,
    float scale,
    bool is_causal,
    bool has_mask,
    int num_heads,
    int num_kv_heads,
    mx::StreamOrDevice s = {});

class Week2DensePrefillMMA : public mx::Primitive {
public:
    Week2DensePrefillMMA(mx::Stream stream, float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          has_mask_(has_mask),
          num_heads_(num_heads),
          num_kv_heads_(num_kv_heads) {}
    // Primitive inputs = {q, k, v, mask}; outputs = {output}.
    // eval_gpu allocates output and follows the launch/buffer contract above.
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("Week2DensePrefillMMA has no vmap implementation.");
    }
    const char *name() const override { return "Week2DensePrefillMMA"; }

private:
    float scale_;
    bool is_causal_;
    bool has_mask_;
    int num_heads_;
    int num_kv_heads_;
};


}  // namespace tiny_llm_ext
