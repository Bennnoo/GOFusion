// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#include "gs/monogs_rasterizer.cuh"

namespace gs {
torch::autograd::tensor_list rasterize_gaussians_mono(
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor sh_dc,
    torch::Tensor sh_rest,
    torch::Tensor colors_precomp,
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3Ds_precomp,
    torch::Tensor theta,
    torch::Tensor rho,
    MonoGaussianRasterizationSettings raster_settings,
    torch::Tensor fisher_means3D_diag,
    torch::Tensor fisher_opacity_diag)
{
    torch::Device device = torch::kCUDA;

    if (!raster_settings.bg.is_cuda()) {
        raster_settings.bg = raster_settings.bg.to(device);
    }

    if (!raster_settings.viewmatrix.is_cuda()) {
        raster_settings.viewmatrix = raster_settings.viewmatrix.to(device);
    }
    if (!raster_settings.projmatrix.is_cuda()) {
        raster_settings.projmatrix = raster_settings.projmatrix.to(device);
    }
    if (!raster_settings.projmatrix_raw.is_cuda()) {
        raster_settings.projmatrix_raw = raster_settings.projmatrix_raw.to(device);
    }
    if (!raster_settings.camera_center.is_cuda()) {
        raster_settings.camera_center = raster_settings.camera_center.to(device);
    }

    means2D = means2D.to(device);
    means3D = means3D.to(device);
    const auto make_empty_float = [&means3D]() {
        return torch::empty({0}, means3D.options());
    };
    const auto make_empty_opacity = [&opacities, &device]() {
        return torch::empty({0}, opacities.options().device(device));
    };

    sh_dc = sh_dc.defined() ? sh_dc.to(device) : make_empty_float();
    sh_rest = sh_rest.defined() ? sh_rest.to(device) : make_empty_float();
    colors_precomp =
        colors_precomp.defined() ? colors_precomp.to(device) : make_empty_float();
    opacities = opacities.to(device);
    scales = scales.to(device);
    rotations = rotations.to(device);
    cov3Ds_precomp =
        cov3Ds_precomp.defined() ? cov3Ds_precomp.to(device) : make_empty_float();
    theta = theta.defined() ? theta.to(device) : make_empty_float();
    rho = rho.defined() ? rho.to(device) : make_empty_float();
    
    fisher_means3D_diag =
        fisher_means3D_diag.defined() ? fisher_means3D_diag.to(device).contiguous()
                                      : make_empty_float();
    fisher_opacity_diag =
        fisher_opacity_diag.defined() ? fisher_opacity_diag.to(device).contiguous()
                                      : make_empty_opacity();

    const bool retain_scratch_for_backward =
        torch::GradMode::is_enabled()
        && (means3D.requires_grad()
            || means2D.requires_grad()
            || sh_dc.requires_grad()
            || sh_rest.requires_grad()
            || colors_precomp.requires_grad()
            || opacities.requires_grad()
            || scales.requires_grad()
            || rotations.requires_grad()
            || cov3Ds_precomp.requires_grad()
            || theta.requires_grad()
            || rho.requires_grad()
            || raster_settings.bg.requires_grad()
            || raster_settings.viewmatrix.requires_grad()
            || raster_settings.projmatrix.requires_grad()
            || raster_settings.projmatrix_raw.requires_grad()
            || raster_settings.camera_center.requires_grad()
            || fisher_means3D_diag.requires_grad()
            || fisher_opacity_diag.requires_grad());

    return _RasterizeGaussiansMono::apply(means3D,
                                          means2D,
                                          sh_dc,
                                          sh_rest,
                                          colors_precomp,
                                          opacities,
                                          scales,
                                          rotations,
                                          cov3Ds_precomp,
                                          theta,
                                          rho,
                                          raster_settings.image_height,
                                          raster_settings.image_width,
                                          raster_settings.tanfovx,
                                          raster_settings.tanfovy,
                                          raster_settings.focal_x,
                                          raster_settings.focal_y,
                                          raster_settings.center_x,
                                          raster_settings.center_y,
                                          raster_settings.render_geometry,
                                          raster_settings.bg,
                                          raster_settings.scale_modifier,
                                          raster_settings.viewmatrix,
                                          raster_settings.projmatrix,
                                          raster_settings.projmatrix_raw,
                                          raster_settings.sh_degree,
                                          raster_settings.camera_center,
                                          raster_settings.prefiltered,
                                          fisher_means3D_diag,
                                          fisher_opacity_diag,
                                          retain_scratch_for_backward);
}


} 
