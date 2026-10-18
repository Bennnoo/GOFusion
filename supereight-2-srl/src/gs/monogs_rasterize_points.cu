/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */
// License copy: supereight-2-srl/mono_cuda_rasterizer/LICENSE.md.


#include "gs/monogs_rasterize_points.cuh"
#include "gs/nvtx_utils.hpp"
#include "gs/raster_scratch_pool.cuh"

#include <torch/torch.h>

#include "mono_config.h"
#include "mono_rasterizer.h"

namespace gsmono {

namespace {

using MonoScratchPool = gs::detail::RasterScratchPool<7>;

MonoScratchPool& monoTrainingScratchPool()
{
    static MonoScratchPool pool;
    return pool;
}

MonoScratchPool& monoPreviewScratchPool()
{
    static MonoScratchPool pool;
    return pool;
}

struct ForwardWorkspaceGuard {
    MonoScratchPool* pool = nullptr;
    int64_t workspace_id = MonoScratchPool::kInvalidSlot;
    bool retain_for_backward = false;
    bool completed = false;

    ~ForwardWorkspaceGuard()
    {
        if (pool) {
            pool->finishForward(workspace_id, completed && retain_for_backward);
        }
    }
};

} 

std::tuple<int,
           int,
           int,
           int,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           int64_t>
RasterizeGaussiansCUDA(const torch::Tensor& background,
                       const torch::Tensor& means3D,
                       const torch::Tensor& colors,
                       const torch::Tensor& opacity,
                       const torch::Tensor& scales,
                       const torch::Tensor& rotations,
                       const float scale_modifier,
                       const torch::Tensor& cov3D_precomp,
                       const torch::Tensor& viewmatrix,
                       const torch::Tensor& projmatrix,
                       const torch::Tensor& projmatrix_raw,
                       const float tan_fovx,
                       const float tan_fovy,
                       const float focal_x,
                       const float focal_y,
                       const float center_x,
                       const float center_y,
                       const int image_height,
                       const int image_width,
                       const torch::Tensor& sh_dc,
                       const torch::Tensor& sh_rest,
                       const int degree,
                       const torch::Tensor& campos,
                       const bool prefiltered,
                       const bool render_geometry,
                       const bool debug,
                       const bool retain_scratch_for_backward)
{
    GOFUSION_NVTX_RANGE("MonoGS/forward_wrapper");
    TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
                "means3D must have dimensions (num_points, 3)");

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, int_opts);
    torch::Tensor n_touched = torch::full({P}, 0, int_opts);
    torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
    torch::Tensor out_opacity = torch::full({1, H, W}, 0.0, float_opts);
    torch::Tensor out_normal = render_geometry
        ? torch::zeros({3, H, W}, float_opts)
        : torch::empty({0}, float_opts);

    auto& scratch_pool = retain_scratch_for_backward ? monoTrainingScratchPool()
                                                     : monoPreviewScratchPool();
    auto scratch = scratch_pool.acquire(means3D.get_device());
    ForwardWorkspaceGuard workspace_guard{
        &scratch_pool, scratch.id, retain_scratch_for_backward, false};
    torch::Tensor geomBuffer = scratch.buffers[0];
    torch::Tensor tileBuffer = scratch.buffers[1];
    torch::Tensor instanceBuffer = scratch.buffers[2];
    torch::Tensor imageFixedBuffer = scratch.buffers[3];
    torch::Tensor bucketBuffer = scratch.buffers[4];
    torch::Tensor geometryPlaneBuffer = scratch.buffers[5];
    torch::Tensor geometryBucketBuffer = scratch.buffers[6];
    const auto& geomFunc = scratch.resize[0];
    const auto& tileFunc = scratch.resize[1];
    const auto& instanceFunc = scratch.resize[2];
    const auto& imageFixedFunc = scratch.resize[3];
    const auto& bucketFunc = scratch.resize[4];
    const auto& geometryPlaneFunc = scratch.resize[5];
    const auto& geometryBucketFunc = scratch.resize[6];

    int rendered = 0;
    int n_buckets = 0;
    int instance_capacity = 0;
    int bucket_capacity = 0;
    if (P != 0) {
        const int M = (sh_dc.size(0) != 0 ? sh_dc.size(1) : 0)
            + (sh_rest.size(0) != 0 ? sh_rest.size(1) : 0);

        rendered = MonoCudaRasterizer::Rasterizer::forward(
            geomFunc,
            tileFunc,
            instanceFunc,
            imageFixedFunc,
            bucketFunc,
            geometryPlaneFunc,
            geometryBucketFunc,
            scratch.reserve_logical[0],
            scratch.reserve_logical[1],
            scratch.logical_capacities[0],
            scratch.logical_capacities[1],
            scratch.summary_device,
            scratch.read_summary,
            P,
            degree,
            M,
            background.contiguous().data_ptr<float>(),
            W,
            H,
            means3D.contiguous().data_ptr<float>(),
            sh_dc.contiguous().data_ptr<float>(),
            sh_rest.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            focal_x,
            focal_y,
            center_x,
            center_y,
            prefiltered,
            render_geometry,
            out_color.contiguous().data_ptr<float>(),
            out_depth.contiguous().data_ptr<float>(),
            out_opacity.contiguous().data_ptr<float>(),
            render_geometry ? out_normal.contiguous().data_ptr<float>() : nullptr,
            radii.contiguous().data_ptr<int>(),
            n_touched.contiguous().data_ptr<int>(),
            &n_buckets,
            &instance_capacity,
            &bucket_capacity,
            debug);
    }

    (void) projmatrix_raw; 
    workspace_guard.completed = true;

    return std::make_tuple(rendered,
                           n_buckets,
                           instance_capacity,
                           bucket_capacity,
                           out_color,
                           radii,
                           geomBuffer,
                           tileBuffer,
                           instanceBuffer,
                           imageFixedBuffer,
                           bucketBuffer,
                           out_depth,
                           out_opacity,
                           n_touched,
                           geometryPlaneBuffer,
                           geometryBucketBuffer,
                           out_normal,
                           retain_scratch_for_backward ? scratch.id
                                                       : MonoScratchPool::kInvalidSlot);
}

void releaseRasterWorkspace(const int64_t workspace_id)
{
    monoTrainingScratchPool().finishBackward(workspace_id);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& background,
                               const torch::Tensor& means3D,
                               const torch::Tensor& radii,
                               const torch::Tensor& colors,
                               const torch::Tensor& scales,
                               const torch::Tensor& rotations,
                               const float scale_modifier,
                               const torch::Tensor& cov3D_precomp,
                               const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix,
                               const torch::Tensor& projmatrix_raw,
                               const float tan_fovx,
                               const float tan_fovy,
                               const float focal_x,
                               const float focal_y,
                               const float center_x,
                               const float center_y,
                               const torch::Tensor& dL_dout_color,
                               const torch::Tensor& dL_dout_depth,
                               const torch::Tensor& dL_dout_opacity,
                               const torch::Tensor& dL_dout_normal,
                               const torch::Tensor& sh_dc,
                               const torch::Tensor& sh_rest,
                               const int degree,
                               const torch::Tensor& campos,
                               const torch::Tensor& rendered_color,
                               const torch::Tensor& rendered_depth,
                               const torch::Tensor& rendered_opacity,
                               const torch::Tensor& rendered_normal,
                               const torch::Tensor& geomBuffer,
                               const int R,
                               const int B,
                               const int R_capacity,
                               const int B_capacity,
                               const torch::Tensor& tileBuffer,
                               const torch::Tensor& instanceBuffer,
                               const torch::Tensor& imageFixedBuffer,
                               const torch::Tensor& bucketBuffer,
                               const torch::Tensor& geometryPlaneBuffer,
                               const torch::Tensor& geometryBucketBuffer,
                               const torch::Tensor& fisher_dL_dmean3D,
                               const torch::Tensor& fisher_dL_dopacity,
                               const bool render_geometry,
                               const bool needs_parameter_grads,
                               const bool debug)
{
    GOFUSION_NVTX_RANGE("MonoGS/backward_wrapper");

    const int P = means3D.size(0);

    const int M = (sh_dc.size(0) != 0 ? sh_dc.size(1) : 0)
        + (sh_rest.size(0) != 0 ? sh_rest.size(1) : 0);

    
    torch::Tensor dL_dmeans3D;
    torch::Tensor dL_dmeans2D;
    torch::Tensor dL_dcolors;
    torch::Tensor dL_ddepths;
    torch::Tensor dL_dplanes;
    torch::Tensor dL_dconic;
    torch::Tensor dL_dopacity;
    torch::Tensor dL_dcov3D;
    torch::Tensor dL_dsh_dc;
    torch::Tensor dL_dsh_rest;
    torch::Tensor dL_dscales;
    torch::Tensor dL_drotations;
    torch::Tensor dL_dtau;
    if (needs_parameter_grads) {
        dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
        dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
        dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
        dL_ddepths = torch::zeros({P, 1}, means3D.options());
        if (render_geometry) {
            dL_dplanes = torch::zeros({P, 4}, means3D.options());
        }
        dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
        dL_dopacity = torch::zeros({P, 1}, means3D.options());
        dL_dcov3D = torch::zeros({P, 6}, means3D.options());
        dL_dsh_dc = torch::zeros(sh_dc.sizes(), means3D.options());
        dL_dsh_rest = torch::zeros(sh_rest.sizes(), means3D.options());
        dL_dscales = torch::zeros({P, 3}, means3D.options());
        dL_drotations = torch::zeros({P, 4}, means3D.options());
        dL_dtau = torch::zeros({P, 6}, means3D.options());
    }
    
    
    if (P != 0) {
        MonoCudaRasterizer::Rasterizer::backward(
            P,
            degree,
            M,
            R,
            B,
            R_capacity,
            B_capacity,
            background.contiguous().data_ptr<float>(),
            dL_dout_color.size(2),
            dL_dout_color.size(1),
            means3D.contiguous().data_ptr<float>(),
            sh_dc.contiguous().data_ptr<float>(),
            sh_rest.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            scales.data_ptr<float>(),
            scale_modifier,
            rotations.data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            projmatrix_raw.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            focal_x,
            focal_y,
            center_x,
            center_y,
            radii.contiguous().data_ptr<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(tileBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(instanceBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageFixedBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(bucketBuffer.contiguous().data_ptr()),
            render_geometry
                ? reinterpret_cast<char*>(geometryPlaneBuffer.contiguous().data_ptr())
                : nullptr,
            render_geometry
                ? reinterpret_cast<char*>(geometryBucketBuffer.contiguous().data_ptr())
                : nullptr,
            rendered_color.contiguous().data_ptr<float>(),
            rendered_depth.contiguous().data_ptr<float>(),
            rendered_opacity.contiguous().data_ptr<float>(),
            render_geometry ? rendered_normal.contiguous().data_ptr<float>() : nullptr,
            dL_dout_color.contiguous().data_ptr<float>(),
            dL_dout_depth.contiguous().data_ptr<float>(),
            dL_dout_opacity.contiguous().data_ptr<float>(),
            render_geometry ? dL_dout_normal.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dmeans2D.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dconic.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dopacity.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dcolors.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_ddepths.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads && render_geometry
                ? dL_dplanes.contiguous().data_ptr<float>()
                : nullptr,
            needs_parameter_grads ? dL_dmeans3D.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dcov3D.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dsh_dc.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dsh_rest.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dscales.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_drotations.contiguous().data_ptr<float>() : nullptr,
            needs_parameter_grads ? dL_dtau.contiguous().data_ptr<float>() : nullptr,
            (fisher_dL_dmean3D.defined() && fisher_dL_dmean3D.numel() > 0)
                ? fisher_dL_dmean3D.contiguous().data_ptr<float>()
                : nullptr,
            (fisher_dL_dopacity.defined() && fisher_dL_dopacity.numel() > 0)
                ? fisher_dL_dopacity.contiguous().data_ptr<float>()
                : nullptr,
            debug);
    }

    return std::make_tuple(dL_dmeans2D,
                           dL_dcolors,
                           dL_dopacity,
                           dL_dmeans3D,
                           dL_dcov3D,
                           dL_dsh_dc,
                           dL_dsh_rest,
                           dL_dscales,
                           dL_drotations,
                           dL_dtau);
}

} 
