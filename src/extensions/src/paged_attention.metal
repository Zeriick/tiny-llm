#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
[[kernel]] void paged_cache_update_kernel(
    device const T* values [[buffer(0)]],
    device T* pages [[buffer(1)]],
    constant const int& heads [[buffer(2)]],
    constant const int& length [[buffer(3)]],
    constant const int& head_dim [[buffer(4)]],
    constant const int& page_size [[buffer(5)]],
    constant const int& page_id [[buffer(6)]],
    constant const int& start [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const int total = heads * length * head_dim;
    if (index >= static_cast<uint>(total)) return;
    const int head_stride = length * head_dim;
    const int head = index / head_stride;
    const int within_head = index - head * head_stride;
    const int token = within_head / head_dim;
    const int dim = within_head - token * head_dim;
    const int destination =
        ((page_id * heads + head) * page_size + start + token) * head_dim + dim;
    pages[destination] = values[index];
}

instantiate_kernel("paged_cache_update_f32", paged_cache_update_kernel, float);
instantiate_kernel("paged_cache_update_bf16", paged_cache_update_kernel, bfloat16_t);


// Week 3, Day 3:
//   paged_cache_update_kernel
// Week 3, Day 4 reference names (the public tests do not require this split):
//   paged_attention_decode
//   paged_attention_scalar_f32
//   paged_attention_scalar_bf16
// Week 3, Day 5:
//   paged_attention_mma_bf16_d128
//
// The public C++ API remains paged_attention across Days 4-5. Day 4 may use one
// correct direct kernel for all shapes; Day 5 optimizes the supported BF16
// long-prefill region behind that stable boundary. Its behavior tests may
// already pass through the Day 4 fallback; implementing and manually tracing
// the reachable optimized region is the learner-owned Day 5 work.
