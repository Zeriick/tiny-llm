#pragma once

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace tiny_llm {

// Copy a row-major device tile into padded threadgroup storage. Each thread
// owns one contiguous source chunk; only the edge path checks individual reads.
template <typename T, int ROWS, int COLS, int DESTINATION_STRIDE, int THREADS, bool TRANSPOSE_DESTINATION = false,
          bool COPY_16_BYTES = false>
struct CooperativeTileLoader {
    static_assert(ROWS > 0 && COLS > 0 && THREADS > 0 && THREADS <= ROWS * COLS);
    static_assert((ROWS * COLS) % THREADS == 0);
    static_assert(COLS % ((ROWS * COLS) / THREADS) == 0);
    static_assert(DESTINATION_STRIDE >= (TRANSPOSE_DESTINATION ? ROWS : COLS));
    static_assert(!COPY_16_BYTES || (!TRANSPOSE_DESTINATION && sizeof(T) * ((ROWS * COLS) / THREADS) == 16));

    struct alignas(sizeof(T)) Read16Bytes {
        uint8_t values[16];
    };

    static METAL_FUNC void load(device const T *source, int source_stride, threadgroup T *destination,
                                uint thread_index, int valid_rows = ROWS, int valid_columns = COLS) {
        if (valid_rows == ROWS && valid_columns == COLS) {
            load_tile<false>(source, source_stride, destination, thread_index, valid_rows, valid_columns);
        } else {
            load_tile<true>(source, source_stride, destination, thread_index, valid_rows, valid_columns);
        }
    }

private:
    template <bool CHECK_BOUNDS>
    static METAL_FUNC void load_tile(device const T *source, int source_stride, threadgroup T *destination,
                                     uint thread_index, int valid_rows, int valid_columns) {
        constexpr int values_per_thread = ROWS * COLS / THREADS;
        const int first = int(thread_index) * values_per_thread;
        const int row = first / COLS;
        const int column = first % COLS;

        if constexpr (COPY_16_BYTES && !CHECK_BOUNDS) {
            *reinterpret_cast<threadgroup Read16Bytes *>(destination + row * DESTINATION_STRIDE + column) =
                *reinterpret_cast<device const Read16Bytes *>(source + row * source_stride + column);
        } else {
            #pragma unroll
            for (int offset = 0; offset < values_per_thread; ++offset) {
                T value;
                if constexpr (CHECK_BOUNDS) {
                    value = row < valid_rows && column + offset < valid_columns
                        ? source[row * source_stride + column + offset] : T(0);
                } else {
                    value = source[row * source_stride + column + offset];
                }
                if constexpr (TRANSPOSE_DESTINATION) {
                    destination[(column + offset) * DESTINATION_STRIDE + row] = value;
                } else {
                    destination[row * DESTINATION_STRIDE + column + offset] = value;
                }
            }
        }
    }
};

// Each lane owns two adjacent columns of an 8x8 fragment. Lane bits 0 and 3
// select the column pair; bits 1, 2, and 4 select the row. Coordinates are (x, y).
METAL_FUNC ushort2 course_matrix_coordinate(ushort lane) {
    return ushort2((lane & 1) * 2 + ((lane >> 3) & 1) * 4,
                   ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4);
}

template <typename T>
METAL_FUNC void course_load_matrix(thread simdgroup_matrix<T, 8, 8> &matrix, threadgroup const T *source,
                                   int row_stride, ushort lane) {
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.y * row_stride + coordinate.x];
    matrix.thread_elements()[1] = source[coordinate.y * row_stride + coordinate.x + 1];
}

template <typename T>
METAL_FUNC void course_load_transposed_matrix(thread simdgroup_matrix<T, 8, 8> &matrix, threadgroup const T *source,
                                              int row_stride, ushort lane) {
    // View [output, reduction] storage as [reduction, output] for A @ W.T.
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.x * row_stride + coordinate.y];
    matrix.thread_elements()[1] = source[(coordinate.x + 1) * row_stride + coordinate.y];
}

// Four SIMD groups cover a 32x32 output tile. Each owns a 16x16 quadrant,
// represented by four FP32 accumulators, and consumes 32 reduction values.
template <typename T, typename OutT, int TILE_STRIDE>
struct CooperativeBlockMMA {
    simdgroup_matrix<float, 8, 8> accumulators[2][2];
    ushort simdgroup_index;
    ushort lane_index;

    METAL_FUNC CooperativeBlockMMA(ushort simdgroup, ushort lane) : simdgroup_index(simdgroup), lane_index(lane) {
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            #pragma unroll
            for (int column = 0; column < 2; ++column) {
                accumulators[row][column].thread_elements() = 0.0f;
            }
        }
    }

    METAL_FUNC void multiply_accumulate(threadgroup const T *left_tile, threadgroup const T *right_tile) {
        const int row_base = (simdgroup_index / 2) * 16;
        const int column_base = (simdgroup_index % 2) * 16;
        #pragma unroll
        for (int reduction = 0; reduction < 32; reduction += 8) {
            simdgroup_matrix<T, 8, 8> left[2];
            simdgroup_matrix<T, 8, 8> right[2];
            #pragma unroll
            for (int operand = 0; operand < 2; ++operand) {
                course_load_matrix(left[operand],
                    left_tile + (row_base + operand * 8) * TILE_STRIDE + reduction,
                    TILE_STRIDE, lane_index);
                course_load_transposed_matrix(right[operand],
                    right_tile + (column_base + operand * 8) * TILE_STRIDE + reduction,
                    TILE_STRIDE, lane_index);
            }
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                #pragma unroll
                for (int column = 0; column < 2; ++column) {
                    simdgroup_multiply_accumulate(accumulators[row][column], left[row], right[column],
                                                 accumulators[row][column]);
                }
            }
        }
    }

    METAL_FUNC void store_result_safe(device OutT *output, int output_stride, short2 valid_shape) const {
        // valid_shape is (columns, rows), relative to this output tile.
        const ushort2 coordinate = course_matrix_coordinate(lane_index);
        const int row_base = (simdgroup_index / 2) * 16 + coordinate.y;
        const int column_base = (simdgroup_index % 2) * 16 + coordinate.x;
        #pragma unroll
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int row = row_base + row_fragment * 8;
            if (row >= valid_shape.y) continue;
            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
                const int column = column_base + column_fragment * 8;
                #pragma unroll
                for (int element = 0; element < 2; ++element) {
                    if (column + element < valid_shape.x) {
                        output[row * output_stride + column + element] =
                            OutT(accumulators[row_fragment][column_fragment].thread_elements()[element]);
                    }
                }
            }
        }
    }
};

}  // namespace tiny_llm
