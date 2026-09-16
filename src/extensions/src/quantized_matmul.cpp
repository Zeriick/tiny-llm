// Copyright © 2024 Apple Inc.

#include "tiny_llm_ext.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

inline void validate_quantized_matmul_inputs(
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int bits,
    const mx::array& a,
    const mx::array& b,
    bool transpose_b) {
    if (a.dtype() != mx::float16 && a.dtype() != mx::bfloat16 && a.dtype() != mx::float32) {
        throw std::runtime_error("quantized_matmul: a must be float16, bfloat16, or float32");
    }
    if (scales.dtype() != a.dtype() || biases.dtype() != a.dtype()) {
        throw std::runtime_error("quantized_matmul: scales/biases must have the same dtype as a");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.ndim() < 2) {
        throw std::runtime_error("quantized_matmul: a must have at least 2 dimensions");
    }
    if (b.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (scales.ndim() != 2 || biases.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: scales and biases must be 2D arrays");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: only 4-bit quantization is supported");
    }
    if (group_size <= 0) {
        throw std::runtime_error("quantized_matmul: group_size must be positive");
    }

    const int packs_per_word = 32 / bits;  // 8 for int4 packed in uint32
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const int64_t k = static_cast<int64_t>(a_shape[a_shape.size() - 1]);

    if (k % group_size != 0) {
        throw std::runtime_error("quantized_matmul: k must be divisible by group_size");
    }
    if (group_size % packs_per_word != 0) {
        throw std::runtime_error("quantized_matmul: group_size must be divisible by packed values per uint32");
    }

    // In this project:
    // - transpose_b=true  => b shape [n, k/8]  (MLX quantize output layout)
    // - transpose_b=false => b shape [k/8, n]
    const int64_t packed_k = transpose_b ? static_cast<int64_t>(b_shape[1]) : static_cast<int64_t>(b_shape[0]);
    const int64_t n = transpose_b ? static_cast<int64_t>(b_shape[0]) : static_cast<int64_t>(b_shape[1]);

    if (k != packed_k * packs_per_word) {
        throw std::runtime_error("quantized_matmul: inner dimension mismatch between a and b");
    }

    const int64_t groups_per_row = k / group_size;
    if (static_cast<int64_t>(scales.shape()[0]) != n || static_cast<int64_t>(scales.shape()[1]) != groups_per_row) {
        throw std::runtime_error("quantized_matmul: scales/biases shape must be [n, k/group_size]");
    }
}

template <typename T>
void quantized_matmul_cpu_impl(
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& a,
    const mx::array& b,
    mx::array& out,
    int group_size,
    int bits,
    bool transpose_b,
    mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<T>(),
                      out_strides = out.strides(),
                      group_size = group_size,
                      bits = bits,
                      transpose_b = transpose_b,
                      a = mx::array::unsafe_weak_copy(a),
                      b = mx::array::unsafe_weak_copy(b),
                      scales = mx::array::unsafe_weak_copy(scales),
                      biases = mx::array::unsafe_weak_copy(biases)]() {
        const T* a_ptr = a.data<T>();
        const uint32_t* b_ptr = b.data<uint32_t>();
        const T* scales_ptr = scales.data<T>();
        const T* biases_ptr = biases.data<T>();

        const auto a_shape = a.shape();
        const auto a_strides = a.strides();
        const auto b_shape = b.shape();
        const auto b_strides = b.strides();
        const auto scales_shape = scales.shape();
        const auto scales_strides = scales.strides();
        const auto biases_shape = biases.shape();
        const auto biases_strides = biases.strides();

        const int a_ndim = static_cast<int>(a_shape.size());
        const int batch_ndim = a_ndim - 2;
        const int64_t m = static_cast<int64_t>(a_shape[a_ndim - 2]);
        const int64_t k = static_cast<int64_t>(a_shape[a_ndim - 1]);
        const int64_t n = transpose_b ? static_cast<int64_t>(b_shape[0]) : static_cast<int64_t>(b_shape[1]);

        const int packs_per_word = 32 / bits;
        const int64_t groups_per_row = k / group_size;
        const int64_t words_per_group = group_size / packs_per_word;
        const int64_t packed_k = k / packs_per_word;
        const uint32_t pack_mask = (1u << bits) - 1u;

        int64_t batch_size = 1;
        for (int axis = 0; axis < batch_ndim; ++axis) {
            batch_size *= static_cast<int64_t>(a_shape[axis]);
        }

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int64_t a_batch_base = 0;
            int64_t out_batch_base = 0;

            int64_t remaining = batch_idx;
            for (int axis = batch_ndim - 1; axis >= 0; --axis) {
                const int64_t axis_size = static_cast<int64_t>(a_shape[axis]);
                const int64_t coord = remaining % axis_size;
                remaining /= axis_size;
                a_batch_base += coord * static_cast<int64_t>(a_strides[axis]);
                out_batch_base += coord * static_cast<int64_t>(out_strides[axis]);
            }

            for (int64_t i = 0; i < m; ++i) {
                const int64_t a_row_base = a_batch_base + i * static_cast<int64_t>(a_strides[batch_ndim]);

                for (int64_t j = 0; j < n; ++j) {
                    float sum = 0.0f;

                    for (int64_t group_idx = 0; group_idx < groups_per_row; ++group_idx) {
                        const int64_t scales_idx = mx::elem_to_loc(
                            j * groups_per_row + group_idx, scales_shape, scales_strides);
                        const int64_t biases_idx = mx::elem_to_loc(
                            j * groups_per_row + group_idx, biases_shape, biases_strides);

                        const float scale = static_cast<float>(scales_ptr[scales_idx]);
                        const float bias = static_cast<float>(biases_ptr[biases_idx]);

                        for (int64_t word_idx = 0; word_idx < words_per_group; ++word_idx) {
                            const int64_t packed_col = group_idx * words_per_group + word_idx;
                            const int64_t b_linear_idx =
                                transpose_b ? (j * packed_k + packed_col) : (packed_col * n + j);
                            const int64_t b_idx = mx::elem_to_loc(b_linear_idx, b_shape, b_strides);
                            const uint32_t packed = b_ptr[b_idx];

                            for (int pack_idx = 0; pack_idx < packs_per_word; ++pack_idx) {
                                const uint32_t q = (packed >> (pack_idx * bits)) & pack_mask;
                                const int64_t k_idx = group_idx * group_size + word_idx * packs_per_word + pack_idx;
                                const int64_t a_idx =
                                    a_row_base + k_idx * static_cast<int64_t>(a_strides[batch_ndim + 1]);
                                const float w = static_cast<float>(q) * scale + bias;
                                sum += static_cast<float>(a_ptr[a_idx]) * w;
                            }
                        }
                    }

                    const int64_t out_idx = out_batch_base + i * static_cast<int64_t>(out_strides[batch_ndim]) +
                                            j * static_cast<int64_t>(out_strides[batch_ndim + 1]);
                    out_ptr[out_idx] = static_cast<T>(sum);
                }
            }
        }
    });
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

mx::array quantized_matmul(
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int bits,
    const mx::array& a,
    const mx::array& b,
    bool transpose_b,
    bool use_simdgroup,
    bool use_split_k,
    mx::StreamOrDevice s) {
    validate_quantized_matmul_inputs(scales, biases, group_size, bits, a, b, transpose_b);

    const auto b_shape = b.shape();
    const int64_t n = transpose_b ? static_cast<int64_t>(b_shape[0]) : static_cast<int64_t>(b_shape[1]);

    auto out_shape = a.shape();
    out_shape[out_shape.size() - 1] = static_cast<size_t>(n);

    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ a.dtype(),
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(
            to_stream(s), group_size, bits, transpose_b, use_simdgroup, use_split_k),
        /* const std::vector<mx::array>& inputs = */ std::vector<mx::array>{scales, biases, a, b});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Implementation
///////////////////////////////////////////////////////////////////////////////

void QuantizedMatmul::eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    auto& scales = inputs[0];
    auto& biases = inputs[1];
    auto& a = inputs[2];
    auto& b = inputs[3];
    auto& out = outputs[0];

    switch (a.dtype()) {
        case mx::float16:
            quantized_matmul_cpu_impl<mx::float16_t>(
                scales, biases, a, b, out, group_size_, bits_, transpose_b_, stream());
            return;
        case mx::float32:
            quantized_matmul_cpu_impl<float>(
                scales, biases, a, b, out, group_size_, bits_, transpose_b_, stream());
            return;
        case mx::bfloat16:
            quantized_matmul_cpu_impl<mx::bfloat16_t>(
                scales, biases, a, b, out, group_size_, bits_, transpose_b_, stream());
            return;
        default:
            throw std::runtime_error("quantized_matmul: unsupported dtype");
    }
}

#ifdef _METAL_

void QuantizedMatmul::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    auto& scales = inputs[0];
    auto& biases = inputs[1];
    auto& a = inputs[2];
    auto& b = inputs[3];
    auto& out = outputs[0];

    if (group_size_ != 128) {
        throw std::runtime_error("quantized_matmul: GPU only supports group_size=128");
    }
    if (bits_ != 4) {
        throw std::runtime_error("quantized_matmul: GPU only supports 4-bit quantization");
    }
    if (a.dtype() != mx::float16 && a.dtype() != mx::bfloat16) {
        throw std::runtime_error("quantized_matmul: GPU only supports float16 or bfloat16");
    }
    if (a.dtype() != scales.dtype() || a.dtype() != biases.dtype()) {
        throw std::runtime_error("quantized_matmul: a, scales, and biases must have the same dtype");
    }

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be row contiguous on GPU");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be row contiguous on GPU");
    }
    if (!scales.flags().row_contiguous || !biases.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: scales and biases must be row contiguous on GPU");
    }

    const int64_t N = static_cast<int64_t>(a.shape().back());
    const int64_t M = static_cast<int64_t>(a.size() / N);
    const int64_t K = static_cast<int64_t>(out.shape().back());

    const int packs_per_word = 32 / bits_;
    if (N % group_size_ != 0 || N % packs_per_word != 0) {
        throw std::runtime_error("quantized_matmul: invalid N for GPU kernel");
    }
    if (static_cast<int64_t>(scales.shape()[0]) != K ||
        static_cast<int64_t>(scales.shape()[1]) != N / group_size_) {
        throw std::runtime_error("quantized_matmul: invalid scales shape for GPU kernel");
    }

    const int64_t expected_b0 = transpose_b_ ? K : (N / packs_per_word);
    const int64_t expected_b1 = transpose_b_ ? (N / packs_per_word) : K;
    if (static_cast<int64_t>(b.shape()[0]) != expected_b0 || static_cast<int64_t>(b.shape()[1]) != expected_b1) {
        throw std::runtime_error("quantized_matmul: invalid b shape for GPU kernel");
    }

    if (M > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() ||
        K > std::numeric_limits<int>::max()) {
        throw std::runtime_error("quantized_matmul: dimensions exceed GPU kernel int range");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto library = d.get_library("tiny_llm_ext");
    if (!transpose_b_) {
        throw std::runtime_error("quantized_matmul: GPU requires transpose_b=True");
    }
    if (use_split_k_) {
        throw std::runtime_error("quantized_matmul: split-K is introduced on Week 2 Day 7");
    }

    const bool use_matvec = use_simdgroup_ && M <= 8;
    const bool use_tiled_matmul = use_simdgroup_ && M > 8;
    const bool is_f16 = a.dtype() == mx::float16;
    const char* kernel_name = use_matvec
        ? (is_f16 ? "quantized_matvec_x4_fast_w4a16_g128_f16"
                  : "quantized_matvec_x4_fast_w4a16_g128_bf16")
        : use_tiled_matmul
        ? (is_f16 ? "quantized_matmul_simdgroup_w4a16_g128_f16"
                  : "quantized_matmul_simdgroup_w4a16_g128_bf16")
        : (is_f16 ? "quantized_matmul_vanilla_w4a16_g128_f16"
                  : "quantized_matmul_vanilla_w4a16_g128_bf16");
    auto kernel = d.get_kernel(kernel_name, library);

    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);

    const int M_i = static_cast<int>(M);
    const int N_i = static_cast<int>(N);
    const int K_i = static_cast<int>(K);
    compute_encoder.set_bytes(M_i, 5);
    compute_encoder.set_bytes(N_i, 6);
    compute_encoder.set_bytes(K_i, 7);

    if (use_matvec) {
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroups_per_threadgroup = 2;
        constexpr int outputs_per_threadgroup =
            outputs_per_simdgroup * simdgroups_per_threadgroup;
        const int column_tiles =
            (static_cast<int>(K) + outputs_per_threadgroup - 1) /
            outputs_per_threadgroup;
        compute_encoder.dispatch_threadgroups(
            MTL::Size(static_cast<size_t>(M) * column_tiles, 1, 1),
            MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    if (use_tiled_matmul) {
        constexpr int tile_size = 32;
        constexpr int simdgroups_per_threadgroup = 4;
        compute_encoder.dispatch_threadgroups(
            MTL::Size((K + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size, 1),
            MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    const size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = M <= 16 ? 16 : 32;
    const int y_size = static_cast<int>(tgp_size) / x_size;
    compute_encoder.dispatch_threadgroups(
        MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1),
        MTL::Size(x_size, y_size, 1));
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

void QuantizedMatmul::print(std::ostream& os) {
    os << name() << "(group_size=" << group_size_ << ", bits=" << bits_ << ", transpose_b=" << transpose_b_ << ")";
}

std::vector<mx::array> QuantizedMatmul::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
    throw std::runtime_error("QuantizedMatmul: JVP not supported");
}

std::vector<mx::array> QuantizedMatmul::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
    throw std::runtime_error("QuantizedMatmul: VJP not supported");
}

std::pair<std::vector<mx::array>, std::vector<int>> QuantizedMatmul::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
    throw std::runtime_error("QuantizedMatmul: vmap not supported");
}

bool QuantizedMatmul::is_equivalent(const mx::Primitive& other) const {
    const QuantizedMatmul& r_other = static_cast<const QuantizedMatmul&>(other);
    return group_size_ == r_other.group_size_ && bits_ == r_other.bits_ &&
           transpose_b_ == r_other.transpose_b_ &&
           use_simdgroup_ == r_other.use_simdgroup_ &&
           use_split_k_ == r_other.use_split_k_;
}

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
