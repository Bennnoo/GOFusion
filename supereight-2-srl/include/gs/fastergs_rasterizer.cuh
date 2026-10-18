// Adapted from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion: C++ integration of the Faster-GS rasterizer.
// License: supereight-2-srl/fastergs_cuda_rasterizer/LICENSE.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)



#pragma once

#include "gs/nvtx_utils.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <utility>

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
forward_wrapper(const torch::Tensor& means,
                const torch::Tensor& scales,
                const torch::Tensor& rotations,
                const torch::Tensor& opacities,
                const torch::Tensor& sh_coefficients_0,
                const torch::Tensor& sh_coefficients_rest,
                const torch::Tensor& w2c,
                const torch::Tensor& cam_position,
                const torch::Tensor& bg_color,
                int active_sh_bases,
                int width,
                int height,
                float focal_x,
                float focal_y,
                float center_x,
                float center_y,
                float near_plane,
                float far_plane,
                bool proper_antialiasing,
                bool render_geometry,
                bool retain_scratch_for_backward);

void release_forward_workspace(int64_t workspace_id);

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
backward_wrapper(torch::Tensor& densification_info,
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
                 int active_sh_bases,
                 int width,
                 int height,
                 float focal_x,
                 float focal_y,
                 float center_x,
                 float center_y,
                 float near_plane,
                 float far_plane,
                 bool proper_antialiasing,
                 bool render_geometry,
                 int n_instances,
                 int n_buckets,
                 int instance_capacity,
                 int bucket_capacity,
                 int instance_primitive_indices_selector);
} 

namespace gs {

struct FasterGaussianRasterizationSettings {
    int image_height;
    int image_width;
    float focal_x;
    float focal_y;
    float center_x;
    float center_y;
    float near_plane;
    float far_plane;
    int active_sh_bases;
    bool proper_antialiasing;
    bool render_geometry = false;
    torch::Tensor bg;
    torch::Tensor viewmatrix;
    torch::Tensor camera_center;
};

torch::autograd::tensor_list rasterize_gaussians_faster(
    torch::Tensor means3D,
    torch::Tensor sh_coefficients_0,
    torch::Tensor sh_coefficients_rest,
    torch::Tensor raw_opacities,
    torch::Tensor raw_scales,
    torch::Tensor raw_rotations,
    torch::Tensor densification_info,
    FasterGaussianRasterizationSettings raster_settings);

class _RasterizeGaussiansFaster : public torch::autograd::Function<_RasterizeGaussiansFaster> {
    public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
                                                torch::Tensor means3D,
                                                torch::Tensor sh_coefficients_0,
                                                torch::Tensor sh_coefficients_rest,
                                                torch::Tensor raw_opacities,
                                                torch::Tensor raw_scales,
                                                torch::Tensor raw_rotations,
                                                torch::Tensor densification_info,
                                                int image_height,
                                                int image_width,
                                                float focal_x,
                                                float focal_y,
                                                float center_x,
                                                float center_y,
                                                float near_plane,
                                                float far_plane,
                                                int active_sh_bases,
                                                bool proper_antialiasing,
                                                bool render_geometry,
                                                torch::Tensor bg,
                                                torch::Tensor viewmatrix,
                                                torch::Tensor camera_center,
                                                bool retain_scratch_for_backward)
    {
        GOFUSION_NVTX_RANGE("FasterGS/autograd_forward");

        bg = bg.contiguous();
        viewmatrix = viewmatrix.contiguous();
        camera_center = camera_center.contiguous();

        auto [image,
              primitive_buffers,
              tile_buffers,
              instance_buffers,
              bucket_buffers,
              geometry_primitive_buffers,
              geometry_bucket_buffers,
              depth,
              alpha,
              normal,
              n_instances,
              n_buckets,
              instance_capacity,
              bucket_capacity,
              instance_primitive_indices_selector,
              workspace_id] =
            faster_gs::rasterization::forward_wrapper(means3D.contiguous(),
                                                      raw_scales.contiguous(),
                                                      raw_rotations.contiguous(),
                                                      raw_opacities.contiguous(),
                                                      sh_coefficients_0.contiguous(),
                                                      sh_coefficients_rest.contiguous(),
                                                      viewmatrix,
                                                      camera_center,
                                                      bg,
                                                      active_sh_bases,
                                                      image_width,
                                                      image_height,
                                                      focal_x,
                                                      focal_y,
                                                      center_x,
                                                      center_y,
                                                      near_plane,
                                                      far_plane,
                                                      proper_antialiasing,
                                                      render_geometry,
                                                      retain_scratch_for_backward);

        torch::Tensor radii = torch::ones({means3D.size(0)},
                                          means3D.options().dtype(torch::kInt32));

        ctx->save_for_backward({image,
                                means3D,
                                raw_scales,
                                raw_rotations,
                                raw_opacities,
                                sh_coefficients_rest,
                                primitive_buffers,
                                tile_buffers,
                                instance_buffers,
                                bucket_buffers,
                                geometry_primitive_buffers,
                                geometry_bucket_buffers,
                                depth,
                                normal,
                                viewmatrix,
                                camera_center,
                                bg,
                                densification_info});
        ctx->saved_data["n_instances"] = n_instances;
        ctx->saved_data["n_buckets"] = n_buckets;
        ctx->saved_data["instance_capacity"] = instance_capacity;
        ctx->saved_data["bucket_capacity"] = bucket_capacity;
        ctx->saved_data["instance_primitive_indices_selector"] =
            instance_primitive_indices_selector;
        ctx->saved_data["image_width"] = image_width;
        ctx->saved_data["image_height"] = image_height;
        ctx->saved_data["focal_x"] = focal_x;
        ctx->saved_data["focal_y"] = focal_y;
        ctx->saved_data["center_x"] = center_x;
        ctx->saved_data["center_y"] = center_y;
        ctx->saved_data["near_plane"] = near_plane;
        ctx->saved_data["far_plane"] = far_plane;
        ctx->saved_data["active_sh_bases"] = active_sh_bases;
        ctx->saved_data["proper_antialiasing"] = proper_antialiasing;
        ctx->saved_data["render_geometry"] = render_geometry;
        ctx->saved_data["workspace_id"] = workspace_id;
        ctx->saved_data["needs_means3D_grad"] = means3D.requires_grad();
        ctx->saved_data["needs_sh_coefficients_0_grad"] = sh_coefficients_0.requires_grad();
        ctx->saved_data["needs_sh_coefficients_rest_grad"] = sh_coefficients_rest.requires_grad();
        ctx->saved_data["needs_opacity_grad"] = raw_opacities.requires_grad();
        ctx->saved_data["needs_scales_grad"] = raw_scales.requires_grad();
        ctx->saved_data["needs_rotations_grad"] = raw_rotations.requires_grad();
        
        return {image, radii, depth, alpha, normal};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        GOFUSION_NVTX_RANGE("FasterGS/autograd_backward");
        struct WorkspaceReleaseGuard {
            int64_t workspace_id;
            ~WorkspaceReleaseGuard()
            {
                faster_gs::rasterization::release_forward_workspace(workspace_id);
            }
        } workspace_release{ctx->saved_data["workspace_id"].toInt()};

        auto grad_out_image = grad_outputs[0].contiguous();

        auto saved = ctx->get_saved_variables();
        auto image = saved[0];
        auto means3D = saved[1];
        auto raw_scales = saved[2];
        auto raw_rotations = saved[3];
        auto raw_opacities = saved[4];
        auto sh_coefficients_rest = saved[5];
        auto primitive_buffers = saved[6];
        auto tile_buffers = saved[7];
        auto instance_buffers = saved[8];
        auto bucket_buffers = saved[9];
        auto geometry_primitive_buffers = saved[10];
        auto geometry_bucket_buffers = saved[11];
        auto depth = saved[12];
        auto normal = saved[13];
        auto viewmatrix = saved[14];
        auto camera_center = saved[15];
        auto bg = saved[16];
        auto densification_info = saved[17];
        
        const bool render_geometry = ctx->saved_data["render_geometry"].toBool();
        const int image_width = ctx->saved_data["image_width"].toInt();
        const int image_height = ctx->saved_data["image_height"].toInt();
        auto grad_out_depth = grad_outputs[2].defined()
            ? grad_outputs[2].contiguous()
            : torch::zeros({1, image_height, image_width}, grad_out_image.options());
        auto grad_out_alpha = grad_outputs[3].defined()
            ? grad_outputs[3].contiguous()
            : torch::zeros({1, image_height, image_width}, grad_out_image.options());
        auto grad_out_normal = render_geometry && grad_outputs[4].defined()
            ? grad_outputs[4].contiguous()
            : torch::zeros({3, image_height, image_width}, grad_out_image.options());

        

auto [grad_means3D,
              grad_scales,
              grad_rotations,
              grad_opacities,
              grad_sh_coefficients_0,
              grad_sh_coefficients_rest] =
            faster_gs::rasterization::backward_wrapper(
                densification_info,
                grad_out_image,
                image,
                grad_out_depth,
                grad_out_alpha,
                grad_out_normal,
                depth,
                normal,
                means3D,
                raw_scales,
                raw_rotations,
                raw_opacities,
                sh_coefficients_rest,
                primitive_buffers,
                tile_buffers,
                instance_buffers,
                bucket_buffers,
                geometry_primitive_buffers,
                geometry_bucket_buffers,
                viewmatrix,
                camera_center,
                bg,
                ctx->saved_data["active_sh_bases"].toInt(),
                ctx->saved_data["image_width"].toInt(),
                ctx->saved_data["image_height"].toInt(),
                static_cast<float>(ctx->saved_data["focal_x"].toDouble()),
                static_cast<float>(ctx->saved_data["focal_y"].toDouble()),
                static_cast<float>(ctx->saved_data["center_x"].toDouble()),
                static_cast<float>(ctx->saved_data["center_y"].toDouble()),
                static_cast<float>(ctx->saved_data["near_plane"].toDouble()),
                static_cast<float>(ctx->saved_data["far_plane"].toDouble()),
                ctx->saved_data["proper_antialiasing"].toBool(),
                render_geometry,
                ctx->saved_data["n_instances"].toInt(),
                ctx->saved_data["n_buckets"].toInt(),
                ctx->saved_data["instance_capacity"].toInt(),
                ctx->saved_data["bucket_capacity"].toInt(),
                ctx->saved_data["instance_primitive_indices_selector"].toInt());

        torch::autograd::tensor_list grads(22);
        grads[0] = ctx->saved_data["needs_means3D_grad"].toBool()
            ? grad_means3D : torch::Tensor();
        grads[1] = ctx->saved_data["needs_sh_coefficients_0_grad"].toBool()
            ? grad_sh_coefficients_0 : torch::Tensor();
        grads[2] = ctx->saved_data["needs_sh_coefficients_rest_grad"].toBool()
            ? grad_sh_coefficients_rest : torch::Tensor();
        grads[3] = ctx->saved_data["needs_opacity_grad"].toBool()
            ? grad_opacities : torch::Tensor();
        grads[4] = ctx->saved_data["needs_scales_grad"].toBool()
            ? grad_scales : torch::Tensor();
        grads[5] = ctx->saved_data["needs_rotations_grad"].toBool()
            ? grad_rotations : torch::Tensor();
        
        return grads;
    }
};

class FasterGaussianRasterizer : torch::nn::Module {
    public:
    explicit FasterGaussianRasterizer(FasterGaussianRasterizationSettings raster_settings) :
            raster_settings_(std::move(raster_settings))
    {
    }

    torch::autograd::tensor_list forward_all(torch::Tensor means3D,
                                                     torch::Tensor raw_opacities,
                                                     torch::Tensor sh_coefficients_0,
                                                     torch::Tensor sh_coefficients_rest,
                                                     torch::Tensor raw_scales,
                                                     torch::Tensor raw_rotations,
                                                     torch::Tensor densification_info =
                                                         torch::Tensor())
    {
        torch::Device device = torch::kCUDA;
        if (!densification_info.defined()) {
            densification_info = torch::empty({0},
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(device));
        }
        
        auto result = rasterize_gaussians_faster(means3D,
                                                 sh_coefficients_0,
                                                 sh_coefficients_rest,
                                                 raw_opacities,
                                                 raw_scales,
                                                 raw_rotations,
                                                 densification_info,
                                                 raster_settings_);
        return result;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor means3D,
        torch::Tensor raw_opacities,
        torch::Tensor sh_coefficients_0,
        torch::Tensor sh_coefficients_rest,
        torch::Tensor raw_scales,
        torch::Tensor raw_rotations,
        torch::Tensor densification_info = torch::Tensor())
    {
        auto result = forward_all(means3D,
                                  raw_opacities,
                                  sh_coefficients_0,
                                  sh_coefficients_rest,
                                  raw_scales,
                                  raw_rotations,
                                  densification_info);
        return {result[0], result[1]};
    }

    private:
    FasterGaussianRasterizationSettings raster_settings_;
};

} 
