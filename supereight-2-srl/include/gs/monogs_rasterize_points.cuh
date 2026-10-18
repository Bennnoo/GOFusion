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


#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace gsmono {

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
                       float scale_modifier,
                       const torch::Tensor& cov3D_precomp,
                       const torch::Tensor& viewmatrix,
                       const torch::Tensor& projmatrix,
                       const torch::Tensor& projmatrix_raw,
                       float tan_fovx,
                       float tan_fovy,
                       float focal_x,
                       float focal_y,
                       float center_x,
                       float center_y,
                       int image_height,
                       int image_width,
                       const torch::Tensor& sh_dc,
                       const torch::Tensor& sh_rest,
                       int degree,
                       const torch::Tensor& campos,
                       bool prefiltered,
                       bool render_geometry,
                       bool debug,
                       bool retain_scratch_for_backward);

void releaseRasterWorkspace(int64_t workspace_id);

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
                               float scale_modifier,
                               const torch::Tensor& cov3D_precomp,
                               const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix,
                               const torch::Tensor& projmatrix_raw,
                               float tan_fovx,
                               float tan_fovy,
                               float focal_x,
                               float focal_y,
                               float center_x,
                               float center_y,
                               const torch::Tensor& dL_dout_color,
                               const torch::Tensor& dL_dout_depth,
                               const torch::Tensor& dL_dout_opacity,
                               const torch::Tensor& dL_dout_normal,
                               const torch::Tensor& sh_dc,
                               const torch::Tensor& sh_rest,
                               int degree,
                               const torch::Tensor& campos,
                               const torch::Tensor& rendered_color,
                               const torch::Tensor& rendered_depth,
                               const torch::Tensor& rendered_opacity,
                               const torch::Tensor& rendered_normal,
                               const torch::Tensor& geomBuffer,
                               int R,
                               int B,
                               int R_capacity,
                               int B_capacity,
                               const torch::Tensor& tileBuffer,
                               const torch::Tensor& instanceBuffer,
                               const torch::Tensor& imageFixedBuffer,
                               const torch::Tensor& bucketBuffer,
                               const torch::Tensor& geometryPlaneBuffer,
                               const torch::Tensor& geometryBucketBuffer,
                               const torch::Tensor& fisher_dL_dmean3D,
                               const torch::Tensor& fisher_dL_dopacity,
                               bool render_geometry,
                               bool needs_parameter_grads,
                               bool debug);

} 
