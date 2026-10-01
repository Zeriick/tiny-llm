#include <algorithm>
#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 3, Day 3. Page storage is request state, so the output aliases the existing page buffer and the Metal kernel touches only the appended slice.
mx::array paged_cache_update(const mx::array &pages, const mx::array &values, int page_id, int start,
                             mx::StreamOrDevice s) {
    if ((pages.dtype() != mx::float32 && pages.dtype() != mx::bfloat16) || values.dtype() != pages.dtype()) {
        throw std::runtime_error("paged_cache_update: pages and values must have the same float32 or bfloat16 dtype");
    }
    if (pages.shape().size() != 4 || values.shape().size() != 4 || values.shape()[0] != 1) {
        throw std::runtime_error(
            "paged_cache_update: expected pages [P, H, page_size, D] and values [1, H, length, D]");
    }
    if (values.shape()[1] != pages.shape()[1] || values.shape()[3] != pages.shape()[3]) {
        throw std::runtime_error("paged_cache_update: values must match the page head count and head dimension");
    }
    if (page_id < 0 || page_id >= pages.shape()[0] || start < 0 || start + values.shape()[2] > pages.shape()[2]) {
        throw std::runtime_error("paged_cache_update: destination slice is outside page storage");
    }
    return mx::array(pages.shape(), pages.dtype(), std::make_shared<PagedCacheUpdate>(to_stream(s), page_id, start),
                     {pages, values});
}

void PagedCacheUpdate::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("paged_cache_update: the course extension is GPU-only");
}

#ifdef _METAL_
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &pages = inputs[0];
    const auto &values = inputs[1];
    auto &out = outputs[0];
    if (!pages.flags().row_contiguous || !values.flags().row_contiguous) {
        throw std::runtime_error("paged_cache_update: pages and values must be contiguous");
    }

    // Page storage is request state, so the output intentionally aliases the
    // input buffer. The kernel touches only the appended slice instead of
    // building a functional copy of the entire cache tensor.
    out.copy_shared_buffer(pages);
    auto &d = mx::metal::device(stream().device);
    auto library = d.get_library("tiny_llm_ext");
    const char *kernel_name = pages.dtype() == mx::bfloat16 ? "paged_cache_update_bf16" : "paged_cache_update_f32";
    auto kernel = d.get_kernel(kernel_name, library);
    auto &compute_encoder = mx::metal::get_command_encoder(stream());
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(values, 0);
    compute_encoder.set_output_array(out, 1);
    const int heads = pages.shape()[1];
    const int length = values.shape()[2];
    const int head_dim = pages.shape()[3];
    const int page_size = pages.shape()[2];
    compute_encoder.set_bytes(heads, 2);
    compute_encoder.set_bytes(length, 3);
    compute_encoder.set_bytes(head_dim, 4);
    compute_encoder.set_bytes(page_size, 5);
    compute_encoder.set_bytes(page_id_, 6);
    compute_encoder.set_bytes(start_, 7);
    const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
    compute_encoder.dispatch_threads(MTL::Size(values.size(), 1, 1), MTL::Size(threads, 1, 1));
}
#else
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("PagedCacheUpdate has no GPU implementation.");
}
#endif

// Week 3, Day 4 owns correct direct page-walking attention for every supported
// shape; one kernel may serve all of them. Day 5 optimizes the supported BF16
// long-prefill region behind this stable API.
mx::array paged_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, const mx::array &,
                          float, bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("paged_attention", "Week 3, Day 4");
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedAttention::eval_cpu", "Week 3, Day 4");
}

void PagedAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedAttention::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
