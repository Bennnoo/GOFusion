// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#include "fastergs/rasterization_api.h"
#include "fastergs/forward.h"
#include "fastergs/backward.h"
#include "fastergs/torch_utils.h"
#include "fastergs/rasterization_config.h"
#include "fastergs/helper_math.h"
#include "gs/nvtx_utils.hpp"
#include "gs/raster_scratch_pool.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <functional>
#include <tuple>

namespace {

using FasterScratchPool = gs::detail::RasterScratchPool<6>;

FasterScratchPool& fasterScratchPool()
{
    static FasterScratchPool pool;
    return pool;
}

struct ForwardWorkspaceGuard {
    FasterScratchPool* pool = nullptr;
    int64_t workspace_id = FasterScratchPool::kInvalidSlot;
    bool retain_for_backward = false;
    bool completed = false;

    ~ForwardWorkspaceGuard()
    {
        if (pool) {
            pool->finishForward(workspace_id, completed && retain_for_backward);
        }
    }
};

} // namespace

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
faster_gs::rasterization::forward_wrapper(
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
    const bool retain_scratch_for_backward)
{
    GOFUSION_NVTX_RANGE("FasterGS/forward_wrapper");

    // all optimizable tensors must be passed as contiguous CUDA float tensors
    CHECK_INPUT(config::debug, means, "means");
    CHECK_INPUT(config::debug, scales, "scales");
    CHECK_INPUT(config::debug, rotations, "rotations");
    CHECK_INPUT(config::debug, opacities, "opacities");
    CHECK_INPUT(config::debug, sh_coefficients_0, "sh_coefficients_0");
    CHECK_INPUT(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor depth = render_geometry
        ? torch::zeros({1, height, width}, float_options)
        : torch::empty({0}, float_options);
    torch::Tensor alpha = render_geometry
        ? torch::zeros({1, height, width}, float_options)
        : torch::empty({0}, float_options);
    torch::Tensor normal = render_geometry
        ? torch::zeros({3, height, width}, float_options)
        : torch::empty({0}, float_options);

    auto& scratch_pool = fasterScratchPool();
    auto scratch = scratch_pool.acquire(means.get_device());
    ForwardWorkspaceGuard workspace_guard{
        &scratch_pool, scratch.id, retain_scratch_for_backward, false};
    torch::Tensor primitive_buffers = scratch.buffers[0];
    torch::Tensor tile_buffers = scratch.buffers[1];
    torch::Tensor instance_buffers = scratch.buffers[2];
    torch::Tensor bucket_buffers = scratch.buffers[3];
    torch::Tensor geometry_primitive_buffers = scratch.buffers[4];
    torch::Tensor geometry_bucket_buffers = scratch.buffers[5];
    const auto& resize_primitive_buffers = scratch.resize[0];
    const auto& resize_tile_buffers = scratch.resize[1];
    const auto& resize_instance_buffers = scratch.resize[2];
    const auto& resize_bucket_buffers = scratch.resize[3];
    const auto& resize_geometry_primitive_buffers = scratch.resize[4];
    const auto& resize_geometry_bucket_buffers = scratch.resize[5];

    auto [n_instances,
          n_buckets,
          instance_capacity,
          bucket_capacity,
          instance_primitive_indices_selector] = forward(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        resize_bucket_buffers,
        resize_geometry_primitive_buffers,
        resize_geometry_bucket_buffers,
        scratch.reserve_logical[0],
        scratch.reserve_logical[1],
        scratch.logical_capacities[0],
        scratch.logical_capacities[1],
        scratch.summary_device,
        scratch.read_summary,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        render_geometry ? depth.data_ptr<float>() : nullptr,
        render_geometry ? alpha.data_ptr<float>() : nullptr,
        render_geometry ? normal.data_ptr<float>() : nullptr,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        render_geometry
    );
    workspace_guard.completed = true;

    return {
        image,
        primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
        geometry_primitive_buffers, geometry_bucket_buffers, depth, alpha, normal,
        n_instances, n_buckets, instance_capacity, bucket_capacity,
        instance_primitive_indices_selector, scratch.id
    };
}

void faster_gs::rasterization::release_forward_workspace(const int64_t workspace_id)
{
    fasterScratchPool().finishBackward(workspace_id);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
faster_gs::rasterization::backward_wrapper(
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
    const int instance_primitive_indices_selector)
{
    GOFUSION_NVTX_RANGE("FasterGS/backward_wrapper");
    c10::cuda::CUDAGuard device_guard(means.device());

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    
    const bool update_densification_info =
        densification_info.defined() && densification_info.numel() > 0;
    const bool needs_param_grads =
        means.requires_grad() || scales.requires_grad() || rotations.requires_grad()
        || opacities.requires_grad() || sh_coefficients_rest.requires_grad();

    torch::Tensor grad_means;
    torch::Tensor grad_scales;
    torch::Tensor grad_rotations;
    torch::Tensor grad_opacities;
    torch::Tensor grad_sh_coefficients_0;
    torch::Tensor grad_sh_coefficients_rest;
    torch::Tensor grad_mean2d_helper;
    torch::Tensor grad_conic_helper;
    torch::Tensor grad_planes;
    if (needs_param_grads || update_densification_info) {
        grad_means = torch::zeros({n_primitives, 3}, float_options);
        grad_scales = torch::zeros({n_primitives, 3}, float_options);
        grad_rotations = torch::zeros({n_primitives, 4}, float_options);
        grad_opacities = torch::zeros({n_primitives, 1}, float_options);
        grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);
        grad_sh_coefficients_rest = torch::zeros({n_primitives, total_sh_bases, 3}, float_options);
        grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
        grad_conic_helper = torch::zeros({3, n_primitives}, float_options);
        if (render_geometry) {
            grad_planes = torch::zeros({n_primitives, 4}, float_options);
        }
    }
    
    

    backward(
        grad_image.data_ptr<float>(),
        image.data_ptr<float>(),
        render_geometry ? grad_depth.data_ptr<float>() : nullptr,
        render_geometry ? grad_alpha.data_ptr<float>() : nullptr,
        render_geometry ? grad_normal.data_ptr<float>() : nullptr,
        render_geometry ? depth.data_ptr<float>() : nullptr,
        render_geometry ? normal.data_ptr<float>() : nullptr,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(tile_buffers.data_ptr()),
        reinterpret_cast<char*>(instance_buffers.data_ptr()),
        reinterpret_cast<char*>(bucket_buffers.data_ptr()),
        render_geometry
            ? reinterpret_cast<char*>(geometry_primitive_buffers.data_ptr())
            : nullptr,
        render_geometry
            ? reinterpret_cast<char*>(geometry_bucket_buffers.data_ptr())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float3*>(grad_means.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float3*>(grad_scales.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float4*>(grad_rotations.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float*>(grad_opacities.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float3*>(grad_sh_coefficients_rest.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info
            ? reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>())
            : nullptr,
        needs_param_grads || update_densification_info ? grad_conic_helper.data_ptr<float>()
                                                       : nullptr,
        needs_param_grads && render_geometry
            ? reinterpret_cast<float4*>(grad_planes.data_ptr<float>())
            : nullptr,
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,
        n_primitives,
        n_instances,
        n_buckets,
        instance_capacity,
        bucket_capacity,
        instance_primitive_indices_selector,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing
    );

    return {grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest};
}
