// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               int,
               int,
               int,
               int,
               int,
               int64_t>
    forward_wrapper(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool render_geometry,
        const bool retain_scratch_for_backward);

    void release_forward_workspace(int64_t workspace_id);

    std::tuple<torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& image,
        const torch::Tensor& grad_depth,
        const torch::Tensor& grad_alpha,
        const torch::Tensor& grad_normal,
        const torch::Tensor& depth,
        const torch::Tensor& normal,
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& primitive_buffers,
        const torch::Tensor& tile_buffers,
        const torch::Tensor& instance_buffers,
        const torch::Tensor& bucket_buffers,
        const torch::Tensor& geometry_primitive_buffers,
        const torch::Tensor& geometry_bucket_buffers,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool render_geometry,
        const int n_instances,
        const int n_buckets,
        const int instance_capacity,
        const int bucket_capacity,
        const int instance_primitive_indices_selector);

}
