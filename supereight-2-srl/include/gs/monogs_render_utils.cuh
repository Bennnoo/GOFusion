// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)

#pragma once

#include <cmath>
#include <torch/torch.h>

#include "gaussian.cuh"
#include "gaussian_utils.cuh"
#include "gs/geometry_supervision.cuh"
#include "gs/monogs_rasterizer.cuh"

namespace gs {

namespace mono_render_detail {
inline bool hasRenderableGaussians(const GaussianModel& gaussianModel)
{
    const auto xyz = gaussianModel.Get_xyz();
    return xyz.defined() && xyz.numel() > 0;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
makeEmptyRenderResult(const Camera& viewpoint_camera)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto image = torch::zeros({3,
                               static_cast<long>(viewpoint_camera.height),
                               static_cast<long>(viewpoint_camera.width)},
                              options)
                     .set_requires_grad(true);
    auto means2D = torch::empty({0, 3}, options).requires_grad_(false);
    auto visibility =
        torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto radii = torch::empty({0}, options);
    return {image, means2D, visibility, radii};
}

inline RenderResult makeEmptyGeometryRenderResult(const Camera& viewpoint_camera)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    RenderResult result;
    result.rgb = torch::zeros({3, viewpoint_camera.height, viewpoint_camera.width}, options)
                     .set_requires_grad(true);
    result.depth_raw = torch::zeros({1, viewpoint_camera.height, viewpoint_camera.width}, options);
    result.alpha = torch::zeros({1, viewpoint_camera.height, viewpoint_camera.width}, options);
    result.normal_raw = torch::zeros({3, viewpoint_camera.height, viewpoint_camera.width}, options);
    result.means2D = torch::empty({0, 3}, options);
    result.visibility = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    result.radii = torch::empty({0}, options);
    return result;
}

inline void setGeometryIntrinsics(MonoGaussianRasterizationSettings& settings,
                                  const Camera& camera)
{
    settings.focal_x = camera.fx > 0.0f
        ? camera.fx
        : static_cast<float>(camera.width) / (2.0f * std::tan(camera.fov_x * 0.5f));
    settings.focal_y = camera.fy > 0.0f
        ? camera.fy
        : static_cast<float>(camera.height) / (2.0f * std::tan(camera.fov_y * 0.5f));
    settings.center_x = camera.cx >= 0.0f ? camera.cx : 0.5f * camera.width;
    settings.center_y = camera.cy >= 0.0f ? camera.cy : 0.5f * camera.height;
    settings.render_geometry = true;
}
} 

inline torch::Tensor ensure_projmatrix_raw_cached(Camera& viewpoint_camera)
{
    const bool needs_refresh =
        !viewpoint_camera.proj_matrix_raw.defined()
        || viewpoint_camera.proj_matrix_raw.numel() != 16
        || viewpoint_camera.proj_matrix_raw.device() != viewpoint_camera.T_W2C.device()
        || viewpoint_camera.proj_matrix_raw.scalar_type() != viewpoint_camera.T_W2C.scalar_type();

    if (needs_refresh) {
        viewpoint_camera.proj_matrix_raw =
            viewpoint_camera.T_W2C.inverse().mm(viewpoint_camera.full_proj_matrix).contiguous();
    } else if (!viewpoint_camera.proj_matrix_raw.is_contiguous()) {
        viewpoint_camera.proj_matrix_raw = viewpoint_camera.proj_matrix_raw.contiguous();
    }

    return viewpoint_camera.proj_matrix_raw;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_mono(Camera& viewpoint_camera,
            GaussianModel& gaussianModel,
            float scaling_modifier = 1.0,
            torch::Tensor override_color = torch::Tensor(),
            MonoGaussianBackwardFisherCapture* fisher_capture = nullptr)
{
    if (!mono_render_detail::hasRenderableGaussians(gaussianModel)) {
        return mono_render_detail::makeEmptyRenderResult(viewpoint_camera);
    }

    torch::Tensor bg_color = gaussianModel.Get_background_cuda();

    MonoGaussianRasterizationSettings raster_settings;
    raster_settings.image_height = static_cast<int>(viewpoint_camera.height);
    raster_settings.image_width = static_cast<int>(viewpoint_camera.width);
    raster_settings.tanfovx = std::tan(viewpoint_camera.fov_x * 0.5f);
    raster_settings.tanfovy = std::tan(viewpoint_camera.fov_y * 0.5f);
    raster_settings.bg = bg_color;
    raster_settings.scale_modifier = scaling_modifier;
    raster_settings.viewmatrix = viewpoint_camera.T_W2C;
    raster_settings.projmatrix = viewpoint_camera.full_proj_matrix;
    raster_settings.projmatrix_raw = ensure_projmatrix_raw_cached(viewpoint_camera);
    raster_settings.sh_degree = gaussianModel.Get_active_sh_degree();
    raster_settings.camera_center = viewpoint_camera.cam_center;
    raster_settings.prefiltered = false;

    MonoGaussianRasterizer rasterizer = MonoGaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(false);
    auto opacity = gaussianModel.Get_opacity();
    auto scales = gaussianModel.Get_scaling();
    auto rotations = gaussianModel.Get_rotation();
    auto shs_dc = torch::Tensor();
    auto shs_rest = torch::Tensor();

    auto colors_precomp = torch::Tensor();
    if (override_color.defined() && override_color.dim() == 2
        && override_color.size(0) == means3D.size(0) && override_color.size(1) == 3) {
        colors_precomp = override_color.to(torch::kCUDA, true).contiguous();
    } else {
        shs_dc = gaussianModel.Get_features_dc_raw();
        shs_rest = gaussianModel.Get_features_rest_raw();
    }
    auto cov3D_precomp = torch::Tensor();

    auto [rendererd_image, radii] = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        shs_dc,
        shs_rest,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp,
        torch::Tensor(),
        torch::Tensor(),
        fisher_capture ? fisher_capture->means3D_diag : torch::Tensor(),
        fisher_capture ? fisher_capture->opacity_diag : torch::Tensor());

    return {rendererd_image, means2D, radii > 0, radii};
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_mono_with_pose(Camera& viewpoint_camera,
                      GaussianModel& gaussianModel,
                      const torch::Tensor& theta,
                      const torch::Tensor& rho,
                      float scaling_modifier = 1.0,
                      torch::Tensor override_color = torch::Tensor(),
                      MonoGaussianBackwardFisherCapture* fisher_capture = nullptr)
{
    if (!mono_render_detail::hasRenderableGaussians(gaussianModel)) {
        return mono_render_detail::makeEmptyRenderResult(viewpoint_camera);
    }

    torch::Tensor bg_color = gaussianModel.Get_background_cuda();

    MonoGaussianRasterizationSettings raster_settings;
    raster_settings.image_height = static_cast<int>(viewpoint_camera.height);
    raster_settings.image_width = static_cast<int>(viewpoint_camera.width);
    raster_settings.tanfovx = std::tan(viewpoint_camera.fov_x * 0.5f);
    raster_settings.tanfovy = std::tan(viewpoint_camera.fov_y * 0.5f);
    raster_settings.bg = bg_color;
    raster_settings.scale_modifier = scaling_modifier;
    raster_settings.viewmatrix = viewpoint_camera.T_W2C;
    raster_settings.projmatrix = viewpoint_camera.full_proj_matrix;
    raster_settings.projmatrix_raw = ensure_projmatrix_raw_cached(viewpoint_camera);
    raster_settings.sh_degree = gaussianModel.Get_active_sh_degree();
    raster_settings.camera_center = viewpoint_camera.cam_center;
    raster_settings.prefiltered = false;

    MonoGaussianRasterizer rasterizer = MonoGaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(false);
    auto opacity = gaussianModel.Get_opacity();
    auto scales = gaussianModel.Get_scaling();
    auto rotations = gaussianModel.Get_rotation();
    auto shs_dc = torch::Tensor();
    auto shs_rest = torch::Tensor();

    auto colors_precomp = torch::Tensor();
    if (override_color.defined() && override_color.dim() == 2
        && override_color.size(0) == means3D.size(0) && override_color.size(1) == 3) {
        colors_precomp = override_color.to(torch::kCUDA, true).contiguous();
    } else {
        shs_dc = gaussianModel.Get_features_dc_raw();
        shs_rest = gaussianModel.Get_features_rest_raw();
    }
    auto cov3D_precomp = torch::Tensor();

    auto [rendererd_image, radii] = rasterizer.forward(means3D,
                                                       means2D,
                                                       opacity,
                                                       shs_dc,
                                                       shs_rest,
                                                       colors_precomp,
                                                       scales,
                                                       rotations,
                                                       cov3D_precomp,
                                                       theta,
                                                       rho,
                                                       fisher_capture ? fisher_capture->means3D_diag
                                                                      : torch::Tensor(),
                                                       fisher_capture ? fisher_capture->opacity_diag
                                                                      : torch::Tensor());

    return {rendererd_image, means2D, radii > 0, radii};
}

inline RenderResult render_mono_geometry(
    Camera& viewpoint_camera,
    GaussianModel& gaussianModel,
    const torch::Tensor& theta = torch::Tensor(),
    const torch::Tensor& rho = torch::Tensor(),
    float scaling_modifier = 1.0f,
    torch::Tensor override_color = torch::Tensor(),
    MonoGaussianBackwardFisherCapture* fisher_capture = nullptr)
{
    if (!mono_render_detail::hasRenderableGaussians(gaussianModel)) {
        return mono_render_detail::makeEmptyGeometryRenderResult(viewpoint_camera);
    }

    MonoGaussianRasterizationSettings raster_settings;
    raster_settings.image_height = static_cast<int>(viewpoint_camera.height);
    raster_settings.image_width = static_cast<int>(viewpoint_camera.width);
    raster_settings.tanfovx = std::tan(viewpoint_camera.fov_x * 0.5f);
    raster_settings.tanfovy = std::tan(viewpoint_camera.fov_y * 0.5f);
    mono_render_detail::setGeometryIntrinsics(raster_settings, viewpoint_camera);
    raster_settings.bg = gaussianModel.Get_background_cuda();
    raster_settings.scale_modifier = scaling_modifier;
    raster_settings.viewmatrix = viewpoint_camera.T_W2C;
    raster_settings.projmatrix = viewpoint_camera.full_proj_matrix;
    raster_settings.projmatrix_raw = ensure_projmatrix_raw_cached(viewpoint_camera);
    raster_settings.sh_degree = gaussianModel.Get_active_sh_degree();
    raster_settings.camera_center = viewpoint_camera.cam_center;
    raster_settings.prefiltered = false;

    MonoGaussianRasterizer rasterizer(raster_settings);
    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(means3D).requires_grad_(false);
    auto shs_dc = torch::Tensor();
    auto shs_rest = torch::Tensor();
    auto colors_precomp = torch::Tensor();
    if (override_color.defined() && override_color.dim() == 2
        && override_color.size(0) == means3D.size(0) && override_color.size(1) == 3) {
        colors_precomp = override_color.to(torch::kCUDA, true).contiguous();
    }
    else {
        shs_dc = gaussianModel.Get_features_dc_raw();
        shs_rest = gaussianModel.Get_features_rest_raw();
    }

    auto outputs = rasterizer.forward_all(
        means3D,
        means2D,
        gaussianModel.Get_opacity(),
        shs_dc,
        shs_rest,
        colors_precomp,
        gaussianModel.Get_scaling(),
        gaussianModel.Get_rotation(),
        torch::Tensor(),
        theta,
        rho,
        fisher_capture ? fisher_capture->means3D_diag : torch::Tensor(),
        fisher_capture ? fisher_capture->opacity_diag : torch::Tensor());

    RenderResult result;
    result.rgb = outputs[0];
    result.radii = outputs[1];
    result.depth_raw = outputs[2];
    result.alpha = outputs[3];
    result.normal_raw = outputs[4];
    result.means2D = means2D;
    result.visibility = result.radii > 0;
    return result;
}





} 
