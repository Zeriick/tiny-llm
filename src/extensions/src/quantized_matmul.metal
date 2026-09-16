#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "cooperative_matrix.h"

template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int row = static_cast<int>(gid.x);
  const int column = static_cast<int>(gid.y);
  if (row >= M || column >= K) return;

  constexpr int bits = 4;
  constexpr int group_size = 128;
  constexpr int values_per_word = 32 / bits;
  constexpr uint32_t mask = (1u << bits) - 1u;
  const int packed_cols = N / values_per_word;
  const int words_per_group = group_size / values_per_word;
  const int groups_per_row = N / group_size;
  float sum = 0.0f;

  for (int group = 0; group < groups_per_row; ++group) {
    const int parameter = column * groups_per_row + group;
    const float scale = static_cast<float>(scales[parameter]);
    const float bias = static_cast<float>(biases[parameter]);
    for (int word = 0; word < words_per_group; ++word) {
      const int packed_col = group * words_per_group + word;
      const uint32_t packed = b[column * packed_cols + packed_col];
      const int activation = row * N + packed_col * values_per_word;
      #pragma clang loop unroll(full)
      for (int value = 0; value < values_per_word; ++value) {
        const float q = static_cast<float>(
            (packed >> (value * bits)) & mask);
        sum += static_cast<float>(a[activation + value]) *
            (q * scale + bias);
      }
    }
  }
  out[row * K + column] = static_cast<T>(sum);
}

// Two SIMD groups share a threadgroup. Each SIMD group computes four output
// columns, while each lane loads two adjacent uint32 words (16 activations).
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 128;
  constexpr int values_per_word = 8;
  constexpr int words_per_lane = 2;
  constexpr int values_per_lane = values_per_word * words_per_lane;
  constexpr int outputs_per_simdgroup = 4;
  constexpr int simdgroups_per_threadgroup = 2;
  constexpr int outputs_per_threadgroup =
      outputs_per_simdgroup * simdgroups_per_threadgroup;

  const int column_tiles =
      (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
  const int row = output_tile / column_tiles;
  const int column_base =
      (output_tile - row * column_tiles) * outputs_per_threadgroup +
      simdgroup * outputs_per_simdgroup;
  if (row >= M || column_base >= K) return;

  const int packed_cols = N / values_per_word;
  const int groups_per_row = N / group_size;
  const int activation_base = row * N;
  float sums[outputs_per_simdgroup] = {0.0f};

  for (int packed_col = lane * words_per_lane;
       packed_col < packed_cols;
       packed_col += 32 * words_per_lane) {
    const int group = packed_col / (group_size / values_per_word);
    float scaled_activations[values_per_lane];
    float activation_sum = 0.0f;
    #pragma clang loop unroll(full)
    for (int word = 0; word < words_per_lane; ++word) {
      const int activation_offset =
          activation_base + (packed_col + word) * values_per_word;
      #pragma clang loop unroll(full)
      for (int value = 0; value < values_per_word; ++value) {
        const int local = word * values_per_word + value;
        const float activation =
            static_cast<float>(a[activation_offset + value]);
        activation_sum += activation;
        scaled_activations[local] = activation /
            static_cast<float>(1 << ((value & 3) * 4));
      }
    }

    #pragma clang loop unroll(full)
    for (int output = 0; output < outputs_per_simdgroup; ++output) {
      const int column = column_base + output;
      if (column >= K) continue;
      const int parameter = column * groups_per_row + group;
      const float scale = static_cast<float>(scales[parameter]);
      const float bias = static_cast<float>(biases[parameter]);
      const device uint16_t* packed =
          reinterpret_cast<const device uint16_t*>(
              b + column * packed_cols + packed_col);
      float quantized_dot = 0.0f;
      #pragma clang loop unroll(full)
      for (int nibble_group = 0;
           nibble_group < values_per_lane / 4;
           ++nibble_group) {
        const uint16_t weights = packed[nibble_group];
        const int local = nibble_group * 4;
        quantized_dot +=
            scaled_activations[local] * (weights & 0x000f) +
            scaled_activations[local + 1] * (weights & 0x00f0) +
            scaled_activations[local + 2] * (weights & 0x0f00) +
            scaled_activations[local + 3] * (weights & 0xf000);
      }
      sums[output] += scale * quantized_dot + bias * activation_sum;
    }
  }

  #pragma clang loop unroll(full)
  for (int output = 0; output < outputs_per_simdgroup; ++output) {
    sums[output] = simd_sum(sums[output]);
  }
  if (lane == 0) {
    #pragma clang loop unroll(full)
    for (int output = 0; output < outputs_per_simdgroup; ++output) {
      const int column = column_base + output;
      if (column < K) {
        out[row * K + column] = static_cast<T>(sums[output]);
      }
    }
  }
}

// M prompt rows, N reduction values, K output columns (the existing ABI).
// Four SIMD groups reuse each pair of 32x32 operand tiles. Weight storage stays
// row-major [output, reduction]; the fragment loader supplies its transposed view.
template <typename T>
[[kernel]] void quantized_matmul_simdgroup_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int tile_size = 32;
  constexpr int tile_stride = 40;  // Pad shared rows to avoid bank conflicts.
  constexpr int group_size = 128;
  constexpr int values_per_word = 8;
  constexpr int words_per_tile_row = tile_size / values_per_word;
  const int row_base = int(tile.y) * tile_size;
  const int column_base = int(tile.x) * tile_size;
  const int valid_rows = min(tile_size, M - row_base);
  const int valid_columns = min(tile_size, K - column_base);
  const int packed_cols = N / values_per_word;
  const int groups_per_row = N / group_size;

  threadgroup T activation_tile[tile_size * tile_stride];
  threadgroup T weight_tile[tile_size * tile_stride];
  threadgroup T group_scales[tile_size];
  threadgroup T group_biases[tile_size];
  using ActivationLoader = tiny_llm::CooperativeTileLoader<
      T, tile_size, tile_size, tile_stride, 128, false, true>;
  tiny_llm::CooperativeBlockMMA<T, T, tile_stride> mma(simdgroup, lane);

  // Four consecutive threads unpack the four uint32 words for one weight row.
  const int weight_row = int(thread_id) / words_per_tile_row;
  const int weight_column = (int(thread_id) % words_per_tile_row) * values_per_word;
  const int output_column = column_base + weight_row;

  for (int group = 0; group < groups_per_row; ++group) {
    // One parameter pair per output column, reused for four 32-value slices.
    if (thread_id < tile_size) {
      const int column = column_base + int(thread_id);
      const int parameter = column * groups_per_row + group;
      group_scales[thread_id] = column < K ? scales[parameter] : T(0);
      group_biases[thread_id] = column < K ? biases[parameter] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = float(group_scales[weight_row]);
    const float bias = float(group_biases[weight_row]);

    // N is divisible by 128, so every reduction slice is a full 32 values.
    for (int slice = 0; slice < group_size; slice += tile_size) {
      const int reduction = group * group_size + slice;
      ActivationLoader::load(a + row_base * N + reduction, N,
                             activation_tile, thread_id, valid_rows, tile_size);
      const uint32_t packed = output_column < K
          ? b[output_column * packed_cols + (reduction + weight_column) / values_per_word]
          : 0;
      #pragma unroll
      for (int value = 0; value < values_per_word; ++value) {
        const float code = float((packed >> (4 * value)) & 0xf);
        weight_tile[weight_row * tile_stride + weight_column + value] =
            T(code * scale + bias);
      }
      // All 128 threads finish loading before any SIMD group reads fragments.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma.multiply_accumulate(activation_tile, weight_tile);
      // All four groups finish reading before the next slice overwrites tiles.
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  mma.store_result_safe(out + row_base * K + column_base, K,
                        short2(valid_columns, valid_rows));
}

instantiate_kernel(
    "quantized_matmul_simdgroup_w4a16_g128_f16",
    quantized_matmul_simdgroup_w4a16_g128,
    half);
instantiate_kernel(
    "quantized_matmul_simdgroup_w4a16_g128_bf16",
    quantized_matmul_simdgroup_w4a16_g128,
    bfloat16_t);
instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_f16",
    quantized_matmul_vanilla_w4a16_g128,
    half);
instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_bf16",
    quantized_matmul_vanilla_w4a16_g128,
    bfloat16_t);
instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_f16",
    quantized_matvec_x4_fast_w4a16_g128,
    half);
instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_bf16",
    quantized_matvec_x4_fast_w4a16_g128,
    bfloat16_t);
// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 3:
//   quantized_matmul_vanilla_w4a16_g128
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 5:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 2, Day 7:
//   quantized_matmul_simdgroup_splitk_w4a16_g128
//   quantized_matmul_splitk_reduce
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.
