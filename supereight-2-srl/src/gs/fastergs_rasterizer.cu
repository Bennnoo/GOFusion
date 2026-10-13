// Adapted from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion: C++ integration of the Faster-GS rasterizer.
// License: supereight-2-srl/fastergs_cuda_rasterizer/LICENSE.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)



#include "gs/fastergs_rasterizer.cuh"

#include "fastergs/rasterization_api.h"

namespace gs {

torch::autograd::tensor_list rasterize_gaussians_faster(
    torch::Tensor means3D,
    torch::Tensor sh_coefficients_0,
    torch::Tensor sh_coefficients_rest,
    torch::Tensor raw_opacities,
    torch::Tensor raw_scales,
    torch::Tensor raw_rotations,
    torch::Tensor densification_info,
    FasterGaussianRasterizationSettings raster_settings)
{
    torch::Device device = torch::kCUDA;

    if (!raster_settings.bg.is_cuda()) {
        raster_settings.bg = raster_settings.bg.to(device);
    }
    if (!raster_settings.viewmatrix.is_cuda()) {
        raster_settings.viewmatrix = raster_settings.viewmatrix.to(device);
    }
    if (!raster_settings.camera_center.is_cuda()) {
        raster_settings.camera_center = raster_settings.camera_center.to(device);
    }

    means3D = means3D.to(device).contiguous();
    sh_coefficients_0 = sh_coefficients_0.to(device).contiguous();
    sh_coefficients_rest = sh_coefficients_rest.to(device).contiguous();
    raw_opacities = raw_opacities.to(device).contiguous();
    raw_scales = raw_scales.to(device).contiguous();
    raw_rotations = raw_rotations.to(device).contiguous();
    densification_info = densification_info.to(device).contiguous();
    

    const bool retain_scratch_for_backward =
        torch::GradMode::is_enabled()
        && (means3D.requires_grad()
            || sh_coefficients_0.requires_grad()
            || sh_coefficients_rest.requires_grad()
            || raw_opacities.requires_grad()
            || raw_scales.requires_grad()
            || raw_rotations.requires_grad()
            || densification_info.requires_grad()
            || raster_settings.bg.requires_grad()
            || raster_settings.viewmatrix.requires_grad()
            || raster_settings.camera_center.requires_grad());

    return _RasterizeGaussiansFaster::apply(means3D,
                                            sh_coefficients_0,
                                            sh_coefficients_rest,
                                            raw_opacities,
                                            raw_scales,
                                            raw_rotations,
                                            densification_info,
                                            raster_settings.image_height,
                                            raster_settings.image_width,
                                            raster_settings.focal_x,
                                            raster_settings.focal_y,
                                            raster_settings.center_x,
                                            raster_settings.center_y,
                                            raster_settings.near_plane,
                                            raster_settings.far_plane,
                                            raster_settings.active_sh_bases,
                                            raster_settings.proper_antialiasing,
                                            raster_settings.render_geometry,
                                            raster_settings.bg,
                                            raster_settings.viewmatrix,
                                            raster_settings.camera_center,
                                            retain_scratch_for_backward);
}

} 
