/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <torch/torch.h>
#include "gs/error_driven_densify_params.hpp"

namespace gs {

struct DensifyMask {
    torch::Tensor mask;          // (H,W) float32 CUDA {0,1}
    torch::Tensor peak_x;        // (K) int64 CUDA, [0, W)
    torch::Tensor peak_y;        // (K) int64 CUDA, [0, H)
};

DensifyMask build_densify_mask(const torch::Tensor& image_detached,
                                        const torch::Tensor& gt_img,
                                        const torch::Tensor& valid_mask,
                                        const ErrorDensifyParams& cfg);

torch::Tensor densify_region_mse(const torch::Tensor& image,
                                 const torch::Tensor& gt_img,
                                 const torch::Tensor& mask);

} // namespace gs
