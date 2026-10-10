#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "cooperative_matrix.h"

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

// Host contract: dim <= 4096, group_size is a full-SIMD multiple and
// dim <= 4 * group_size. Keep input values live across the row reduction.
template <typename T>
[[kernel]] void week2_rms_norm_register_cached(
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
    constexpr uint values_per_thread = 4;
    float cached[values_per_thread];
    float sum = 0.0f;
    const size_t row_start = static_cast<size_t>(row) * dim;
    const uint groups = group_size / simd_width;

    #pragma unroll
    for (uint j = 0; j < values_per_thread; ++j) {
        const uint col = thread_index + j * group_size;
        cached[j] = col < static_cast<uint>(dim) ? static_cast<float>(x[row_start + col]) : 0.0f;
        sum += cached[j] * cached[j];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All lanes participate, including threads with no valid input element.
    if (simdgroup == 0) {
        float group_sum = lane < groups ? partial_sums[lane] : 0.0f;
        group_sum = simd_sum(group_sum);
        if (lane == 0) {
            partial_sums[0] = group_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);
    #pragma unroll
    for (uint j = 0; j < values_per_thread; ++j) {
        const uint col = thread_index + j * group_size;
        if (col < static_cast<uint>(dim)) {
            out[row_start + col] = static_cast<T>(cached[j] * inverse_rms * static_cast<float>(weight[col]));
        }
    }
}

instantiate_kernel("week2_rms_norm_register_cached_f32", week2_rms_norm_register_cached, float);
instantiate_kernel("week2_rms_norm_register_cached_f16", week2_rms_norm_register_cached, half);
instantiate_kernel("week2_rms_norm_register_cached_bf16", week2_rms_norm_register_cached, bfloat16_t);

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

// Week 2, Day 6 adds week2_decode_attention.

// Day 5 helpers and tiled dense prefill.
namespace {
// An 8x8 fragment assigns each lane two neighboring columns of one row.
// XOR 1 and XOR 8 exchange with the other three lanes of that SAME row.
// A full simd_sum here would incorrectly mix eight independent query rows.

constant constexpr int DENSE_MMA_SIZE = 8;

template <int N>
inline float dense_row_max(
    thread const simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE>* matrices) {
    float value = -INFINITY;
    for (int fragment_idx = 0; fragment_idx < N; ++fragment_idx) {
        const auto elements = matrices[fragment_idx].thread_elements();
        value = max(value, max(elements[0], elements[1]));
    }
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    value = max(value, simd_shuffle_xor(value, ushort(8)));
    return value;
}

template <int N>
inline float dense_row_sum(
    thread const simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE>* matrices) {
    float value = 0.0f;
    for (int fragment_idx = 0; fragment_idx < N; ++fragment_idx) {
        const auto elements = matrices[fragment_idx].thread_elements();
        value += elements[0] + elements[1];
    }
    value += simd_shuffle_xor(value, ushort(1));
    value += simd_shuffle_xor(value, ushort(8));
    return value;
}

inline void dense_clear_matrix(
    thread simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE>& matrix) {
    matrix.thread_elements()[0] = 0.0f;
    matrix.thread_elements()[1] = 0.0f;
}

inline void dense_scale_matrix_rows(
    thread simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE>& matrix,
    float scale) {
    matrix.thread_elements()[0] *= scale;
    matrix.thread_elements()[1] *= scale;
}

template <typename T>
inline void dense_matrix_multiply_accumulate(
    thread simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE>& accumulator,
    thread simdgroup_matrix<T, DENSE_MMA_SIZE, DENSE_MMA_SIZE>& left,
    thread simdgroup_matrix<T, DENSE_MMA_SIZE, DENSE_MMA_SIZE>& right) {
    simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE> result;
    simdgroup_multiply_accumulate(result, left, right, accumulator);
    accumulator = result;
}

}  // namespace

[[kernel, max_total_threads_per_threadgroup(128)]] void week2_dense_prefill_mma_bf16_d128(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant const int& length [[buffer(5)]],
    constant const int& context [[buffer(6)]],
    constant const float& scale [[buffer(7)]],
    constant const uint& is_causal [[buffer(8)]],
    constant const uint& has_mask [[buffer(9)]],
    constant const int& num_heads [[buffer(10)]],
    constant const int& num_kv_heads [[buffer(11)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HEAD_DIM = 128;
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int SIMD_GROUPS = 4;
    constexpr int THREADS = SIMD_GROUPS * 32;
    constexpr int LDQ = HEAD_DIM + 2;
    constexpr int LDK = BK + 2;
    constexpr int LDV = HEAD_DIM + 2;
    constexpr int KV_STORAGE = HEAD_DIM * LDK;
    constexpr int SCORE_FRAGMENTS = BK / DENSE_MMA_SIZE;
    constexpr int OUTPUT_FRAGMENTS = HEAD_DIM / DENSE_MMA_SIZE;
    using QBlockLoader = tiny_llm::CooperativeTileLoader<
        bfloat, BQ, HEAD_DIM, LDQ, THREADS>;
    using KBlockLoader = tiny_llm::CooperativeTileLoader<
        bfloat, BK, HEAD_DIM, LDK, THREADS, true>;
    using VBlockLoader = tiny_llm::CooperativeTileLoader<
        bfloat, BK, HEAD_DIM, LDV, THREADS>;

    const int query_block = group_id.x;
    const int query_row_group = group_id.y;
    const int batch = query_row_group / num_heads;
    const int query_head = query_row_group - batch * num_heads;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int kv_row = batch * num_kv_heads + kv_head;
    const int thread_idx = simd_gid * 32 + lane;
    const ushort2 coordinate = tiny_llm::course_matrix_coordinate(lane);
    const int query_row_in_block = simd_gid * DENSE_MMA_SIZE + coordinate.y;
    const int query_position = query_block * BQ + query_row_in_block;
    const bool query_valid = query_position < length;
    const int live_queries = clamp(length - query_block * BQ, 0, BQ);

    // 1. Stage Q once. K is transposed in shared memory and its storage is
    // reused for V. 32*130 + 128*18 BF16 values = 12,928 bytes.
    threadgroup bfloat q_tile[BQ * LDQ];
    threadgroup bfloat kv_tile[KV_STORAGE];
    QBlockLoader::load(
        q + size_t(query_row_group) * length * HEAD_DIM + query_block * BQ * HEAD_DIM,
        HEAD_DIM,
        q_tile,
        thread_idx,
        live_queries,
        HEAD_DIM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE> output[OUTPUT_FRAGMENTS];
    for (int fragment_idx = 0; fragment_idx < OUTPUT_FRAGMENTS; ++fragment_idx) {
        dense_clear_matrix(output[fragment_idx]);
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const int total_tiles = (context + BK - 1) / BK;
    int tile_limit = total_tiles;
    if (is_causal) {
        const int last_query = min((query_block + 1) * BQ, length) - 1;
        const int last_key = last_query + (context - length);
        tile_limit = clamp((last_key + 1 + BK - 1) / BK, 0, total_tiles);
    }

    for (int tile = 0; tile < tile_limit; ++tile) {
        const int tile_start = tile * BK;
        const int live_keys = clamp(context - tile_start, 0, BK);
        // 2. Stage K, then compute the 8x16 scores owned by each SIMD group.
        KBlockLoader::load(
            k + (size_t(kv_row) * context + tile_start) * HEAD_DIM,
            HEAD_DIM,
            kv_tile,
            thread_idx,
            live_keys,
            HEAD_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, DENSE_MMA_SIZE, DENSE_MMA_SIZE> scores[SCORE_FRAGMENTS];
        for (int fragment_idx = 0; fragment_idx < SCORE_FRAGMENTS; ++fragment_idx) {
            dense_clear_matrix(scores[fragment_idx]);
        }
        for (int dim = 0; dim < HEAD_DIM; dim += DENSE_MMA_SIZE) {
            simdgroup_matrix<bfloat, DENSE_MMA_SIZE, DENSE_MMA_SIZE> q_fragment;
            tiny_llm::course_load_matrix(
                q_fragment,
                q_tile + simd_gid * DENSE_MMA_SIZE * LDQ + dim,
                LDQ,
                lane);
            for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; ++key_fragment) {
                simdgroup_matrix<bfloat, DENSE_MMA_SIZE, DENSE_MMA_SIZE> k_fragment;
                tiny_llm::course_load_matrix(
                    k_fragment,
                    kv_tile + dim * LDK + key_fragment * DENSE_MMA_SIZE,
                    LDK,
                    lane);
                dense_matrix_multiply_accumulate(scores[key_fragment], q_fragment, k_fragment);
            }
        }

        // 3. Scale scores and apply source tails, causal bounds and additive mask.
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; ++key_fragment) {
            thread auto& values = scores[key_fragment].thread_elements();
            for (int element = 0; element < 2; ++element) {
                const int tile_key = key_fragment * DENSE_MMA_SIZE + coordinate.x + element;
                const int key_position = tile_start + tile_key;
                bool valid = query_valid && key_position < context;
                if (is_causal) {
                    valid = valid && key_position <= query_position + (context - length);
                }
                float score = valid ? values[element] * scale : -INFINITY;
                if (valid && has_mask) {
                    score += mask[(size_t(query_row_group) * length + query_position) * context + key_position];
                }
                values[element] = score;
            }
        }

        // 4. Online softmax: rescale BOTH running denominator and numerator.
        const float tile_max = dense_row_max<SCORE_FRAGMENTS>(scores);
        const float new_max = max(running_max, tile_max);
        const bool finite_row = query_valid && new_max != -INFINITY;
        const float previous_scale = running_max == -INFINITY || !finite_row
            ? 0.0f
            : exp(running_max - new_max);
        for (int fragment_idx = 0; fragment_idx < SCORE_FRAGMENTS; ++fragment_idx) {
            thread auto& values = scores[fragment_idx].thread_elements();
            for (int element = 0; element < 2; ++element) {
                values[element] = values[element] == -INFINITY || !finite_row
                    ? 0.0f
                    : exp(values[element] - new_max);
            }
        }
        const float tile_sum = dense_row_sum<SCORE_FRAGMENTS>(scores);
        running_max = new_max;
        running_sum = previous_scale * running_sum + tile_sum;
        for (int fragment_idx = 0; fragment_idx < OUTPUT_FRAGMENTS; ++fragment_idx) {
            dense_scale_matrix_rows(output[fragment_idx], previous_scale);
        }

        // 5. BF16 operands feed matrix hardware; m, l and output accumulate in
        // FP32. This probability conversion adds BF16 rounding (as in the course).
        simdgroup_matrix<bfloat, DENSE_MMA_SIZE, DENSE_MMA_SIZE> probabilities[SCORE_FRAGMENTS];
        for (int fragment_idx = 0; fragment_idx < SCORE_FRAGMENTS; ++fragment_idx) {
            const auto values = scores[fragment_idx].thread_elements();
            probabilities[fragment_idx].thread_elements()[0] = bfloat(values[0]);
            probabilities[fragment_idx].thread_elements()[1] = bfloat(values[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        VBlockLoader::load(
            v + (size_t(kv_row) * context + tile_start) * HEAD_DIM,
            HEAD_DIM,
            kv_tile,
            thread_idx,
            live_keys,
            HEAD_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; ++output_fragment) {
            for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; ++key_fragment) {
                simdgroup_matrix<bfloat, DENSE_MMA_SIZE, DENSE_MMA_SIZE> value_fragment;
                tiny_llm::course_load_matrix(
                    value_fragment,
                    kv_tile + key_fragment * DENSE_MMA_SIZE * LDV + output_fragment * DENSE_MMA_SIZE,
                    LDV,
                    lane);
                dense_matrix_multiply_accumulate(
                    output[output_fragment], probabilities[key_fragment], value_fragment);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. Normalize once. Fully masked rows have zero mass and output zero.
    if (query_valid) {
        for (int fragment_idx = 0; fragment_idx < OUTPUT_FRAGMENTS; ++fragment_idx) {
            const auto values = output[fragment_idx].thread_elements();
            for (int element = 0; element < 2; ++element) {
                const int dim = fragment_idx * DENSE_MMA_SIZE + coordinate.x + element;
                out[(size_t(query_row_group) * length + query_position) * HEAD_DIM + dim] = running_sum == 0.0f
                    ? bfloat(0.0f)
                    : bfloat(values[element] / running_sum);
            }
        }
    }
}
