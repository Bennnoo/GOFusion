// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.

#pragma once

#include <torch/torch.h>
#include <functional>

inline std::function<char*(size_t N)> resize_function_wrapper(torch::Tensor& t) {
    auto lambda = [&t](const size_t N) {
        t.resize_({static_cast<long long>(N)});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

#define CHECK_INPUT(debug, tensor, name) \
if (debug) { \
if (!tensor.is_cuda() || !tensor.is_floating_point() || !tensor.is_contiguous()) { \
throw std::runtime_error("Input tensor '" + std::string(name) + "' must be a contiguous CUDA float tensor."); \
} \
}
