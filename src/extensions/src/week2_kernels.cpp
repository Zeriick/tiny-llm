#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

void require_float_dtype(const mx::array &x, const char *name) {
    if (x.dtype() != mx::float32 && x.dtype() != mx::float16 && x.dtype() != mx::bfloat16) {
        throw std::runtime_error(std::string(name) + ": expected float32, float16, or bfloat16");
    }
}

const char *dtype_suffix(const mx::array &x) {
    if (x.dtype() == mx::float32) {
        return "f32";
    }
    if (x.dtype() == mx::float16) {
        return "f16";
    }
    if (x.dtype() == mx::bfloat16) {
        return "bf16";
    }
    throw std::runtime_error("unsupported dtype");
}

template <typename T>
void rms_norm_cpu_impl(const mx::array &x, const mx::array &weight, mx::array &out, float eps, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(weight);
    encoder.set_output_array(out);

    const auto shape = x.shape();
    const auto strides = x.strides();
    const auto weight_stride = weight.strides()[0];
    const size_t dim = shape.back();
    const size_t rows = x.size() / dim;

    encoder.dispatch([x_ptr = x.data<T>(), weight_ptr = weight.data<T>(), out_ptr = out.data<T>(), shape, strides,
                      weight_stride, dim, rows, eps]() {
        for (size_t row = 0; row < rows; ++row) {
            float sum_sq = 0.0f;
            for (size_t col = 0; col < dim; ++col) {
                const size_t logical_index = row * dim + col;
                const auto x_offset = mx::elem_to_loc(logical_index, shape, strides);
                const float value = static_cast<float>(x_ptr[x_offset]);
                sum_sq += value * value;
            }

            const float inverse_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
            for (size_t col = 0; col < dim; ++col) {
                const size_t logical_index = row * dim + col;
                const auto x_offset = mx::elem_to_loc(logical_index, shape, strides);
                const auto weight_offset = static_cast<size_t>(col * weight_stride);
                out_ptr[logical_index] = static_cast<T>(static_cast<float>(x_ptr[x_offset]) * inverse_rms *
                                                        static_cast<float>(weight_ptr[weight_offset]));
            }
        }
    });
}

template <typename T>
void rope_cpu_impl(const mx::array &x, const mx::array &offsets, mx::array &out, int dims, float base, bool traditional,
                   mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(offsets);
    encoder.set_output_array(out);

    const auto shape = x.shape();
    const auto strides = x.strides();
    const auto offsets_stride = offsets.strides()[0];
    const size_t batch = shape[0];
    const size_t length = shape[1];
    const size_t heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t half_dims = static_cast<size_t>(dims / 2);

    encoder.dispatch([x_ptr = x.data<T>(), offsets_ptr = offsets.data<int32_t>(), out_ptr = out.data<T>(), shape,
                      strides, offsets_stride, batch, length, heads, head_dim, half_dims, dims, base, traditional]() {
        for (size_t b = 0; b < batch; ++b) {
            const int32_t offset = offsets_ptr[static_cast<size_t>(b * offsets_stride)];
            for (size_t position = 0; position < length; ++position) {
                for (size_t head = 0; head < heads; ++head) {
                    const size_t row_base = ((b * length + position) * heads + head) * head_dim;

                    // Dimensions after `dims` pass through unchanged.
                    for (size_t d = dims; d < head_dim; ++d) {
                        const size_t logical_index = row_base + d;
                        const auto x_offset = mx::elem_to_loc(logical_index, shape, strides);
                        out_ptr[logical_index] = x_ptr[x_offset];
                    }
                }

                // A position and pair share the same rotation across all heads.
                for (size_t pair = 0; pair < half_dims; ++pair) {
                    const float exponent = -static_cast<float>(pair) / static_cast<float>(half_dims);
                    const float angle =
                        (static_cast<float>(offset) + static_cast<float>(position)) * std::pow(base, exponent);
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);

                    const size_t real_dim = traditional ? pair * 2 : pair;
                    const size_t imag_dim = traditional ? real_dim + 1 : pair + half_dims;
                    for (size_t head = 0; head < heads; ++head) {
                        const size_t row_base = ((b * length + position) * heads + head) * head_dim;
                        const size_t real_index = row_base + real_dim;
                        const size_t imag_index = row_base + imag_dim;
                        const auto real_offset = mx::elem_to_loc(real_index, shape, strides);
                        const auto imag_offset = mx::elem_to_loc(imag_index, shape, strides);
                        const float real = static_cast<float>(x_ptr[real_offset]);
                        const float imag = static_cast<float>(x_ptr[imag_offset]);

                        out_ptr[real_index] = static_cast<T>(real * cosine - imag * sine);
                        out_ptr[imag_index] = static_cast<T>(imag * cosine + real * sine);
                    }
                }
            }
        }
    });
}

template <typename T>
void swiglu_cpu_impl(const mx::array &gate, const mx::array &up, mx::array &out, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(gate);
    encoder.set_input_array(up);
    encoder.set_output_array(out);

    const auto shape = gate.shape();
    const auto gate_strides = gate.strides();
    const auto up_strides = up.strides();
    const size_t size = out.size();

    encoder.dispatch([gate_ptr = gate.data<T>(), up_ptr = up.data<T>(), out_ptr = out.data<T>(), shape, gate_strides,
                      up_strides, size]() {
        for (size_t index = 0; index < size; ++index) {
            const auto gate_offset = mx::elem_to_loc(index, shape, gate_strides);
            const auto up_offset = mx::elem_to_loc(index, shape, up_strides);
            const float gate_value = static_cast<float>(gate_ptr[gate_offset]);
            const float silu = gate_value / (1.0f + std::exp(-gate_value));
            out_ptr[index] = static_cast<T>(silu * static_cast<float>(up_ptr[up_offset]));
        }
    });
}

}  // namespace

// Week 2, Day 4.
mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s) {
    require_float_dtype(x, "rms_norm");
    if (x.ndim() == 0 || x.shape().back() == 0) {
        throw std::runtime_error("rms_norm: x must have a non-empty final dimension");
    }
    if (x.dtype() != weight.dtype() || weight.ndim() != 1 || weight.shape()[0] != x.shape().back()) {
        throw std::runtime_error("rms_norm: weight must match the input dtype and final dimension");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RMSNorm>(to_stream(s), eps), {x, weight});
}

mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s) {
    require_float_dtype(x, "rope");
    if (x.ndim() != 4 || offsets.dtype() != mx::int32 || offsets.ndim() != 1 || offsets.shape()[0] != x.shape()[0]) {
        throw std::runtime_error("rope: expected x=[B,L,H,D] and one int32 offset per batch row");
    }
    if (dims <= 0 || dims > x.shape()[3] || dims % 2 != 0) {
        throw std::runtime_error("rope: dims must be positive, even, and no larger than the head dimension");
    }
    if (!std::isfinite(base) || base <= 0.0f) {
        throw std::runtime_error("rope: base must be finite and positive");
    }
    // The Metal kernel indexes the native B,L,H,D layout directly.
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RoPE>(to_stream(s), dims, base, traditional),
                     {mx::contiguous(x, false, s), mx::contiguous(offsets, false, s)});
}

mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s) {
    require_float_dtype(gate, "swiglu");
    if (gate.dtype() != up.dtype() || gate.shape() != up.shape()) {
        throw std::runtime_error("swiglu: gate and up must have the same shape and dtype");
    }
    return mx::array(gate.shape(), gate.dtype(), std::make_shared<Week2SwiGLU>(to_stream(s)),
                     {mx::contiguous(gate, false, s), mx::contiguous(up, false, s)});
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];

    if (out.dtype() == mx::float32) {
        return rms_norm_cpu_impl<float>(x, weight, out, eps_, stream());
    }
    if (out.dtype() == mx::float16) {
        return rms_norm_cpu_impl<mx::float16_t>(x, weight, out, eps_, stream());
    }
    if (out.dtype() == mx::bfloat16) {
        return rms_norm_cpu_impl<mx::bfloat16_t>(x, weight, out, eps_, stream());
    }
    throw std::runtime_error("rms_norm: unsupported dtype");
}

#ifdef _METAL_

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &device = mx::metal::device(stream().device);
    auto kernel =
        device.get_kernel(std::string("week2_rms_norm_") + dtype_suffix(out), device.get_library("tiny_llm_ext"));

    constexpr int threads_per_threadgroup = 256;
    const auto simd_width = kernel->threadExecutionWidth();
    if (simd_width == 0 || threads_per_threadgroup % simd_width != 0) {
        throw std::runtime_error("rms_norm: threadgroup size must be a multiple of the SIMD width");
    }
    if (threads_per_threadgroup > kernel->maxTotalThreadsPerThreadgroup()) {
        throw std::runtime_error("rms_norm: threadgroup size exceeds the kernel limit");
    }
    const auto groups = threads_per_threadgroup / simd_width;
    // The second reduction assigns at most one partial sum to each SIMD lane.
    assert(groups <= simd_width);
    if (groups > simd_width) {
        // Keep the check active in release builds, where assert is disabled.
        throw std::runtime_error("rms_norm: SIMD group count must not exceed the SIMD width");
    }

    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);

    const int rows = static_cast<int>(x.size() / x.shape().back());
    const int dim = static_cast<int>(x.shape().back());
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(dim, 4);
    encoder.set_bytes(eps_, 5);

    // Metal requires threadgroup memory lengths to be multiples of 16 bytes.
    const auto partial_sums_bytes = groups * sizeof(float);
    const auto threadgroup_memory_bytes = (partial_sums_bytes + 15) / 16 * 16;
    encoder.set_threadgroup_memory_length(threadgroup_memory_bytes, 0);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads_per_threadgroup, 1, 1));
}

#else

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rms_norm: Metal unavailable");
}

#endif

void Week2RoPE::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &offsets = inputs[1];
    auto &out = outputs[0];

    if (out.size() == 0) {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        return;
    }

    if (out.dtype() == mx::float32) {
        return rope_cpu_impl<float>(x, offsets, out, dims_, base_, traditional_, stream());
    }
    if (out.dtype() == mx::float16) {
        return rope_cpu_impl<mx::float16_t>(x, offsets, out, dims_, base_, traditional_, stream());
    }
    if (out.dtype() == mx::bfloat16) {
        return rope_cpu_impl<mx::bfloat16_t>(x, offsets, out, dims_, base_, traditional_, stream());
    }
    throw std::runtime_error("rope: unsupported dtype");
}

#ifdef _METAL_

void Week2RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &offsets = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
        return;
    }

    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel(std::string("week2_rope_") + dtype_suffix(out), device.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(offsets, 1);
    encoder.set_output_array(out, 2);

    const int length = x.shape()[1];
    const int heads = x.shape()[2];
    const int head_dim = x.shape()[3];
    const float log2_base = std::log2(base_);
    encoder.set_bytes(length, 3);
    encoder.set_bytes(heads, 4);
    encoder.set_bytes(head_dim, 5);
    encoder.set_bytes(dims_, 6);
    encoder.set_bytes(log2_base, 7);
    encoder.set_bytes(traditional_, 8);

    // Grid axes are pair, block of four heads, and flattened batch/token.
    const size_t pairs = dims_ / 2;
    const size_t head_blocks = (static_cast<size_t>(heads) + 3) / 4;
    const size_t tokens = static_cast<size_t>(x.shape()[0]) * length;
    const size_t group_size = std::min({pairs, size_t(256), kernel->maxTotalThreadsPerThreadgroup()});
    encoder.dispatch_threads(MTL::Size(pairs, head_blocks, tokens), MTL::Size(group_size, 1, 1));
}

#else

void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rope: Metal unavailable");
}

#endif

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];
    const auto &up = inputs[1];
    auto &out = outputs[0];

    if (out.size() == 0) {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        return;
    }

    if (out.dtype() == mx::float32) {
        return swiglu_cpu_impl<float>(gate, up, out, stream());
    }
    if (out.dtype() == mx::float16) {
        return swiglu_cpu_impl<mx::float16_t>(gate, up, out, stream());
    }
    if (out.dtype() == mx::bfloat16) {
        return swiglu_cpu_impl<mx::bfloat16_t>(gate, up, out, stream());
    }
    throw std::runtime_error("swiglu: unsupported dtype");
}

#ifdef _METAL_

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];
    const auto &up = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const size_t size = out.size();
    if (size == 0) {
        return;
    }

    auto &device = mx::metal::device(stream().device);
    auto kernel =
        device.get_kernel(std::string("week2_swiglu_") + dtype_suffix(out), device.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(gate, 0);
    encoder.set_input_array(up, 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(size, 3);

    // One thread loads both branches and writes one fused result.
    const size_t group_size = std::min({size, size_t(256), kernel->maxTotalThreadsPerThreadgroup()});
    encoder.dispatch_threads(MTL::Size(size, 1, 1), MTL::Size(group_size, 1, 1));
}

#else

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("swiglu: Metal unavailable");
}

#endif

// Week 2, Day 5.
mx::array decode_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, float, bool,
                           bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("decode_attention", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_cpu", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_gpu", "Week 2, Day 5");
}

}  // namespace tiny_llm_ext
