// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)

#pragma once

#include <cmath>
#include <torch/torch.h>

#include "fastergs_rasterizer.cuh"
#include "gaussian.cuh"
#include "gaussian_utils.cuh"
#include "gs/geometry_supervision.cuh"

namespace gs {
namespace faster_render_detail {
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
} 

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render(Camera& viewpoint_camera,
       GaussianModel& gaussianModel)
{
    if (!faster_render_detail::hasRenderableGaussians(gaussianModel)) {
        return faster_render_detail::makeEmptyRenderResult(viewpoint_camera);
    }

    torch::Tensor bg_color = gaussianModel.Get_background_cuda();

    const float default_fx =
        static_cast<float>(viewpoint_camera.width) / (2.0f * std::tan(viewpoint_camera.fov_x * 0.5f));
    const float default_fy =
        static_cast<float>(viewpoint_camera.height) / (2.0f * std::tan(viewpoint_camera.fov_y * 0.5f));
    const float focal_x = viewpoint_camera.fx > 0.0f ? viewpoint_camera.fx : default_fx;
    const float focal_y = viewpoint_camera.fy > 0.0f ? viewpoint_camera.fy : default_fy;
    const float center_x =
        viewpoint_camera.cx >= 0.0f ? viewpoint_camera.cx : 0.5f * static_cast<float>(viewpoint_camera.width);
    const float center_y =
        viewpoint_camera.cy >= 0.0f ? viewpoint_camera.cy : 0.5f * static_cast<float>(viewpoint_camera.height);

    const torch::Tensor& w2c_row_major = fixedPoseW2CRowMajor(viewpoint_camera);

    FasterGaussianRasterizationSettings raster_settings;
    raster_settings.image_height = static_cast<int>(viewpoint_camera.height);
    raster_settings.image_width = static_cast<int>(viewpoint_camera.width);
    raster_settings.focal_x = focal_x;
    raster_settings.focal_y = focal_y;
    raster_settings.center_x = center_x;
    raster_settings.center_y = center_y;
    raster_settings.near_plane = 0.2f;
    raster_settings.far_plane = 1000.0f;
    raster_settings.bg = bg_color;
    raster_settings.viewmatrix = w2c_row_major;
    const int sh_degree = gaussianModel.Get_active_sh_degree();
    raster_settings.active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
    raster_settings.camera_center = viewpoint_camera.cam_center;
    raster_settings.proper_antialiasing = false;

    FasterGaussianRasterizer rasterizer = FasterGaussianRasterizer(raster_settings);
    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(false);
    auto opacities_raw = gaussianModel.Get_opacity_raw();
    auto scales_raw = gaussianModel.Get_scaling_raw();
    auto rotations_raw = gaussianModel.Get_rotation_raw();
    auto sh_coeff_0 = gaussianModel.Get_features_dc_raw();
    auto sh_coeff_rest = gaussianModel.Get_features_rest_raw();

    auto [rendererd_image, radii] = rasterizer.forward(
        means3D,
        opacities_raw,
        sh_coeff_0,
        sh_coeff_rest,
        scales_raw,
        rotations_raw);

    return {rendererd_image, means2D, radii > 0, radii};
}

inline RenderResult render_faster_geometry(Camera& viewpoint_camera,
                                           GaussianModel& gaussianModel)
{
    if (!faster_render_detail::hasRenderableGaussians(gaussianModel)) {
        return faster_render_detail::makeEmptyGeometryRenderResult(viewpoint_camera);
    }

    const float default_fx = static_cast<float>(viewpoint_camera.width)
        / (2.0f * std::tan(viewpoint_camera.fov_x * 0.5f));
    const float default_fy = static_cast<float>(viewpoint_camera.height)
        / (2.0f * std::tan(viewpoint_camera.fov_y * 0.5f));
    FasterGaussianRasterizationSettings settings;
    settings.image_height = viewpoint_camera.height;
    settings.image_width = viewpoint_camera.width;
    settings.focal_x = viewpoint_camera.fx > 0.0f ? viewpoint_camera.fx : default_fx;
    settings.focal_y = viewpoint_camera.fy > 0.0f ? viewpoint_camera.fy : default_fy;
    settings.center_x = viewpoint_camera.cx >= 0.0f
        ? viewpoint_camera.cx : 0.5f * viewpoint_camera.width;
    settings.center_y = viewpoint_camera.cy >= 0.0f
        ? viewpoint_camera.cy : 0.5f * viewpoint_camera.height;
    settings.near_plane = 0.2f;
    settings.far_plane = 1000.0f;
    settings.bg = gaussianModel.Get_background_cuda();
    settings.viewmatrix = fixedPoseW2CRowMajor(viewpoint_camera);
    const int sh_degree = gaussianModel.Get_active_sh_degree();
    settings.active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
    settings.camera_center = viewpoint_camera.cam_center;
    settings.proper_antialiasing = false;
    settings.render_geometry = true;

    FasterGaussianRasterizer rasterizer(settings);
    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(means3D).requires_grad_(false);
    auto outputs = rasterizer.forward_all(means3D,
                                          gaussianModel.Get_opacity_raw(),
                                          gaussianModel.Get_features_dc_raw(),
                                          gaussianModel.Get_features_rest_raw(),
                                          gaussianModel.Get_scaling_raw(),
                                          gaussianModel.Get_rotation_raw());
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
