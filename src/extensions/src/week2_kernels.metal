#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Week 2, Day 4: fused RMSNorm.
// Each threadgroup owns one logical input row. Its SIMD groups reduce the
// row in parallel, then all threads perform the final normalization pass.
template <typename T>
[[kernel]] void week2_rms_norm(
    device const T* x [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& dim [[buffer(4)]],
    constant const float& eps [[buffer(5)]],
    threadgroup float* partial_sums [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    // The host launches a one-dimensional threadgroup with complete SIMD groups.
    const int threads_per_threadgroup = static_cast<int>(group_size);
    const uint groups = group_size / simd_width;

    // First reduction: each SIMD group reduces the strided elements assigned
    // to its lanes and leaves one partial sum in threadgroup memory.
    float sum = 0.0f;
    for (int col = static_cast<int>(thread_index); col < dim; col += threads_per_threadgroup) {
        const float value = static_cast<float>(x[row * dim + col]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[simdgroup] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second reduction: SIMD group zero reduces the partial sums. The host
    // asserts groups <= simd_width so each lane loads at most one partial sum.
    if (simdgroup == 0) {
        float group_sum = lane < groups ? partial_sums[lane] : 0.0f;
        group_sum = simd_sum(group_sum);
        if (lane == 0) {
            partial_sums[0] = group_sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);
    for (int col = static_cast<int>(thread_index); col < dim; col += threads_per_threadgroup) {
        const float value = static_cast<float>(x[row * dim + col]);
        out[row * dim + col] = static_cast<T>(value * inverse_rms * static_cast<float>(weight[col]));
    }
}

instantiate_kernel("week2_rms_norm_f32", week2_rms_norm, float);
instantiate_kernel("week2_rms_norm_f16", week2_rms_norm, half);
instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);

// Native B,L,H,D RoPE: each thread rotates one pair in up to four heads.
template <typename T, bool fast_math>
[[kernel]] void week2_rope(
    device const T* x [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& length [[buffer(3)]],
    constant const int& heads [[buffer(4)]],
    constant const int& head_dim [[buffer(5)]],
    constant const int& dims [[buffer(6)]],
    constant const float& log2_base [[buffer(7)]],
    constant const bool& traditional [[buffer(8)]],
    uint3 index [[thread_position_in_grid]]) {
    const uint half_dims = static_cast<uint>(dims / 2);
    const uint pair = index.x;
    const uint first_head = index.y * 4;
    const uint batch = index.z / static_cast<uint>(length);
    const uint position = index.z % static_cast<uint>(length);
    const float exponent = -static_cast<float>(pair) / static_cast<float>(half_dims) * log2_base;
    const float absolute_position = static_cast<float>(offsets[batch]) + static_cast<float>(position);

    float cosine;
    float sine;
    if constexpr (fast_math) {
        const float angle = absolute_position * fast::exp2(exponent);
        cosine = fast::cos(angle);
        sine = fast::sin(angle);
    } else {
        const float angle = absolute_position * exp2(exponent);
        cosine = cos(angle);
        sine = sin(angle);
    }

    const uint real_dim = traditional ? pair * 2 : pair;
    const uint imag_dim = traditional ? real_dim + 1 : pair + half_dims;
    for (uint h = 0; h < 4; ++h) {
        const uint head = first_head + h;
        if (head >= static_cast<uint>(heads)) {
            break;
        }
        const size_t row = (static_cast<size_t>(index.z) * heads + head) * head_dim;
        const float real = static_cast<float>(x[row + real_dim]);
        const float imag = static_cast<float>(x[row + imag_dim]);
        out[row + real_dim] = static_cast<T>(real * cosine - imag * sine);
        out[row + imag_dim] = static_cast<T>(imag * cosine + real * sine);

        // Distribute unrotated dimensions across the pair threads as well.
        for (uint d = static_cast<uint>(dims) + pair; d < static_cast<uint>(head_dim); d += half_dims) {
            out[row + d] = x[row + d];
        }
    }
}

instantiate_kernel("week2_rope_f32", week2_rope, float, false);
instantiate_kernel("week2_rope_f16", week2_rope, half, false);
instantiate_kernel("week2_rope_bf16", week2_rope, bfloat16_t, true);

// One thread owns one element of both branches; intermediates stay in float.
template <typename T>
[[kernel]] void week2_swiglu(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const size_t& size [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= size) {
        return;
    }
    const float gate_value = static_cast<float>(gate[index]);
    const float up_value = static_cast<float>(up[index]);
    const float silu = gate_value / (1.0f + exp(-gate_value));
    out[index] = static_cast<T>(silu * up_value);
}

instantiate_kernel("week2_swiglu_f32", week2_swiglu, float);
instantiate_kernel("week2_swiglu_f16", week2_swiglu, half);
instantiate_kernel("week2_swiglu_bf16", week2_swiglu, bfloat16_t);

// Week 2, Day 5 adds week2_decode_attention.
