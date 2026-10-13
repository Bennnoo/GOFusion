// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)

#pragma once

#include <utility>

#include "gs/monogs_rasterize_points.cuh"
#include "gs/nvtx_utils.hpp"

namespace gs {

struct MonoGaussianRasterizationSettings {
    int image_height;
    int image_width;
    float tanfovx;
    float tanfovy;
    float focal_x = 0.0f;
    float focal_y = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    torch::Tensor bg;
    float scale_modifier;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    torch::Tensor projmatrix_raw;
    int sh_degree;
    torch::Tensor camera_center;
    bool prefiltered;
    bool render_geometry = false;
};

;

struct MonoGaussianBackwardFisherCapture {
    torch::Tensor means3D_diag;
    torch::Tensor opacity_diag;
};

torch::autograd::tensor_list rasterize_gaussians_mono(torch::Tensor means3D,
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
                                                      MonoGaussianRasterizationSettings
                                                          raster_settings,
                                                      torch::Tensor fisher_means3D_diag =
                                                          torch::Tensor(),
                                                      torch::Tensor fisher_opacity_diag =
                                                          torch::Tensor());

class _RasterizeGaussiansMono : public torch::autograd::Function<_RasterizeGaussiansMono> {
    public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
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
                                                int image_height,
                                                int image_width,
                                                float tanfovx,
                                                float tanfovy,
                                                float focal_x,
                                                float focal_y,
                                                float center_x,
                                                float center_y,
                                                bool render_geometry,
                                                torch::Tensor bg,
                                                float scale_modifier,
                                                torch::Tensor viewmatrix,
                                                torch::Tensor projmatrix,
                                                torch::Tensor projmatrix_raw,
                                                int sh_degree,
                                                torch::Tensor camera_center,
                                                bool prefiltered,
                                                torch::Tensor fisher_means3D_diag,
                                                torch::Tensor fisher_opacity_diag,
                                                bool retain_scratch_for_backward)
    {
        GOFUSION_NVTX_RANGE("MonoGS/autograd_forward");
        (void) means2D;

        if (!(focal_x > 0.0f)) {
            focal_x = static_cast<float>(image_width) / (2.0f * tanfovx);
        }
        if (!(focal_y > 0.0f)) {
            focal_y = static_cast<float>(image_height) / (2.0f * tanfovy);
        }

        camera_center = camera_center.contiguous();

        auto [num_rendered,
              num_buckets,
              instance_capacity,
              bucket_capacity,
              color,
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
              workspace_id] = gsmono::RasterizeGaussiansCUDA(bg,
                                                          means3D,
                                                          colors_precomp,
                                                          opacities,
                                                          scales,
                                                          rotations,
                                                          scale_modifier,
                                                          cov3Ds_precomp,
                                                          viewmatrix,
                                                          projmatrix,
                                                          projmatrix_raw,
                                                          tanfovx,
                                                          tanfovy,
                                                          focal_x,
                                                          focal_y,
                                                          center_x,
                                                          center_y,
                                                          image_height,
                                                          image_width,
                                                          sh_dc,
                                                          sh_rest,
                                                          sh_degree,
                                                          camera_center,
                                                          prefiltered,
                                                          render_geometry,
                                                          false,
                                                          retain_scratch_for_backward);

        (void) n_touched;

        ctx->save_for_backward({colors_precomp,
                                means3D,
                                scales,
                                rotations,
                                cov3Ds_precomp,
                                radii,
                                sh_dc,
                                sh_rest,
                                geomBuffer,
                                tileBuffer,
                                instanceBuffer,
                                imageFixedBuffer,
                                bucketBuffer,
                                color,
                                out_depth,
                                out_opacity,
                                out_normal,
                                geometryPlaneBuffer,
                                geometryBucketBuffer});
        const bool has_pose_params =
            theta.defined() && rho.defined() && theta.numel() == 3 && rho.numel() == 3;

        ctx->saved_data["num_rendered"] = num_rendered;
        ctx->saved_data["num_buckets"] = num_buckets;
        ctx->saved_data["instance_capacity"] = instance_capacity;
        ctx->saved_data["bucket_capacity"] = bucket_capacity;
        ctx->saved_data["workspace_id"] = workspace_id;
        ctx->saved_data["has_pose_params"] = has_pose_params;
        ctx->saved_data["background"] = bg;
        ctx->saved_data["scale_modifier"] = scale_modifier;
        ctx->saved_data["viewmatrix"] = viewmatrix;
        ctx->saved_data["projmatrix"] = projmatrix;
        ctx->saved_data["projmatrix_raw"] = projmatrix_raw;
        ctx->saved_data["tanfovx"] = tanfovx;
        ctx->saved_data["tanfovy"] = tanfovy;
        ctx->saved_data["focal_x"] = focal_x;
        ctx->saved_data["focal_y"] = focal_y;
        ctx->saved_data["center_x"] = center_x;
        ctx->saved_data["center_y"] = center_y;
        ctx->saved_data["render_geometry"] = render_geometry;
        ctx->saved_data["image_height"] = image_height;
        ctx->saved_data["image_width"] = image_width;
        ctx->saved_data["sh_degree"] = sh_degree;
        ctx->saved_data["camera_center"] = camera_center;
        ctx->saved_data["prefiltered"] = prefiltered;
        ctx->saved_data["needs_means3D_grad"] = means3D.requires_grad();
        ctx->saved_data["needs_means2D_grad"] = means2D.requires_grad();
        ctx->saved_data["needs_sh_dc_grad"] = sh_dc.requires_grad();
        ctx->saved_data["needs_sh_rest_grad"] = sh_rest.requires_grad();
        ctx->saved_data["needs_colors_precomp_grad"] = colors_precomp.requires_grad();
        ctx->saved_data["needs_opacity_grad"] = opacities.requires_grad();
        ctx->saved_data["needs_scales_grad"] = scales.requires_grad();
        ctx->saved_data["needs_rotations_grad"] = rotations.requires_grad();
        ctx->saved_data["needs_cov3D_grad"] = cov3Ds_precomp.requires_grad();
        
        ctx->saved_data["fisher_means3D_diag"] = fisher_means3D_diag;
        ctx->saved_data["fisher_opacity_diag"] = fisher_opacity_diag;
        return {color, radii, out_depth, out_opacity, out_normal};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        GOFUSION_NVTX_RANGE("MonoGS/autograd_backward");
        struct WorkspaceReleaseGuard {
            int64_t workspace_id;
            ~WorkspaceReleaseGuard()
            {
                gsmono::releaseRasterWorkspace(workspace_id);
            }
        } workspace_release{ctx->saved_data["workspace_id"].toInt()};

        auto grad_out_color = grad_outputs[0];

        auto num_rendered = ctx->saved_data["num_rendered"].to<int>();
        auto saved = ctx->get_saved_variables();
        auto colors_precomp = saved[0];
        auto means3D = saved[1];
        auto scales = saved[2];
        auto rotations = saved[3];
        auto cov3Ds_precomp = saved[4];
        auto radii = saved[5];
        auto sh_dc = saved[6];
        auto sh_rest = saved[7];
        auto geomBuffer = saved[8];
        auto tileBuffer = saved[9];
        auto instanceBuffer = saved[10];
        auto imageFixedBuffer = saved[11];
        auto bucketBuffer = saved[12];
        auto rendered_color = saved[13];
        auto rendered_depth = saved[14];
        auto rendered_opacity = saved[15];
        auto rendered_normal = saved[16];
        auto geometryPlaneBuffer = saved[17];
        auto geometryBucketBuffer = saved[18];
        
        auto fisher_xyz_diag = ctx->saved_data["fisher_means3D_diag"].to<torch::Tensor>();
        auto fisher_opacity_diag = ctx->saved_data["fisher_opacity_diag"].to<torch::Tensor>();
        const int image_height = ctx->saved_data["image_height"].to<int>();
        const int image_width = ctx->saved_data["image_width"].to<int>();
        const bool render_geometry = ctx->saved_data["render_geometry"].toBool();
        auto grad_out_depth = grad_outputs[2].defined()
            ? grad_outputs[2].contiguous()
            : torch::zeros({1, image_height, image_width}, grad_out_color.options());
        auto grad_out_opacity = grad_outputs[3].defined()
            ? grad_outputs[3].contiguous()
            : torch::zeros({1, image_height, image_width}, grad_out_color.options());
        auto grad_out_normal = render_geometry && grad_outputs[4].defined()
            ? grad_outputs[4].contiguous()
            : torch::zeros({3, image_height, image_width}, grad_out_color.options());

auto [grad_means2D,
              grad_colors_precomp,
              grad_opacities,
              grad_means3D,
              grad_cov3Ds_precomp,
              grad_sh_dc,
              grad_sh_rest,
              grad_scales,
              grad_rotations,
              grad_tau] = gsmono::RasterizeGaussiansBackwardCUDA(
            ctx->saved_data["background"].to<torch::Tensor>(),
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            ctx->saved_data["scale_modifier"].to<float>(),
            cov3Ds_precomp,
            ctx->saved_data["viewmatrix"].to<torch::Tensor>(),
            ctx->saved_data["projmatrix"].to<torch::Tensor>(),
            ctx->saved_data["projmatrix_raw"].to<torch::Tensor>(),
            ctx->saved_data["tanfovx"].to<float>(),
            ctx->saved_data["tanfovy"].to<float>(),
            ctx->saved_data["focal_x"].to<float>(),
            ctx->saved_data["focal_y"].to<float>(),
            ctx->saved_data["center_x"].to<float>(),
            ctx->saved_data["center_y"].to<float>(),
            grad_out_color,
            grad_out_depth,
            grad_out_opacity,
            grad_out_normal,
            sh_dc,
            sh_rest,
            ctx->saved_data["sh_degree"].to<int>(),
            ctx->saved_data["camera_center"].to<torch::Tensor>(),
            rendered_color,
            rendered_depth,
            rendered_opacity,
            rendered_normal,
            geomBuffer,
            num_rendered,
            ctx->saved_data["num_buckets"].to<int>(),
            ctx->saved_data["instance_capacity"].to<int>(),
            ctx->saved_data["bucket_capacity"].to<int>(),
            tileBuffer,
            instanceBuffer,
            imageFixedBuffer,
            bucketBuffer,
            geometryPlaneBuffer,
            geometryBucketBuffer,
            fisher_xyz_diag,
            fisher_opacity_diag,
            render_geometry,
            ctx->saved_data["needs_means3D_grad"].toBool()
                || ctx->saved_data["needs_means2D_grad"].toBool()
                || ctx->saved_data["needs_sh_dc_grad"].toBool()
                || ctx->saved_data["needs_sh_rest_grad"].toBool()
                || ctx->saved_data["needs_colors_precomp_grad"].toBool()
                || ctx->saved_data["needs_opacity_grad"].toBool()
                || ctx->saved_data["needs_scales_grad"].toBool()
                || ctx->saved_data["needs_rotations_grad"].toBool()
                || ctx->saved_data["needs_cov3D_grad"].toBool()
                || ctx->saved_data["has_pose_params"].toBool(),
            false);

        torch::Tensor grad_theta;
        torch::Tensor grad_rho;
        if (ctx->saved_data["has_pose_params"].toBool()) {
            auto grad_tau_sum = grad_tau.reshape({-1, 6}).sum(0).contiguous();
            grad_rho = grad_tau_sum.slice(0, 0, 3).contiguous();
            grad_theta = grad_tau_sum.slice(0, 3, 6).contiguous();
        }

        torch::autograd::tensor_list grads(31);
        grads[0] = ctx->saved_data["needs_means3D_grad"].toBool()
            ? grad_means3D : torch::Tensor();
        grads[1] = ctx->saved_data["needs_means2D_grad"].toBool()
            ? grad_means2D : torch::Tensor();
        grads[2] = ctx->saved_data["needs_sh_dc_grad"].toBool()
            ? grad_sh_dc : torch::Tensor();
        grads[3] = ctx->saved_data["needs_sh_rest_grad"].toBool()
            ? grad_sh_rest : torch::Tensor();
        grads[4] = ctx->saved_data["needs_colors_precomp_grad"].toBool()
            ? grad_colors_precomp : torch::Tensor();
        grads[5] = ctx->saved_data["needs_opacity_grad"].toBool()
            ? grad_opacities : torch::Tensor();
        grads[6] = ctx->saved_data["needs_scales_grad"].toBool()
            ? grad_scales : torch::Tensor();
        grads[7] = ctx->saved_data["needs_rotations_grad"].toBool()
            ? grad_rotations : torch::Tensor();
        grads[8] = ctx->saved_data["needs_cov3D_grad"].toBool()
            ? grad_cov3Ds_precomp : torch::Tensor();
        grads[9] = grad_theta;
        grads[10] = grad_rho;
        
        return grads;
    }
};

;

class MonoGaussianRasterizer : torch::nn::Module {
    public:
    MonoGaussianRasterizer(MonoGaussianRasterizationSettings raster_settings) :
            raster_settings_(std::move(raster_settings))
    {
    }

    torch::autograd::tensor_list forward_all(torch::Tensor means3D,
                                                      torch::Tensor means2D,
                                                      torch::Tensor opacities,
                                                      torch::Tensor shs_dc = torch::Tensor(),
                                                      torch::Tensor shs_rest = torch::Tensor(),
                                                      torch::Tensor colors_precomp =
                                                          torch::Tensor(),
                                                      torch::Tensor scales = torch::Tensor(),
                                                      torch::Tensor rotations = torch::Tensor(),
                                                      torch::Tensor cov3D_precomp =
                                                          torch::Tensor(),
                                                      torch::Tensor theta = torch::Tensor(),
                                                      torch::Tensor rho = torch::Tensor(),
                                                      torch::Tensor fisher_means3D_diag =
                                                          torch::Tensor(),
                                                      torch::Tensor fisher_opacity_diag =
                                                          torch::Tensor())
    {
        if (shs_dc.defined() != shs_rest.defined()) {
            throw std::invalid_argument(
                "Please provide SH DC and SH rest together!");
        }
        const bool has_sh = shs_dc.defined() && shs_rest.defined();
        if ((has_sh && colors_precomp.defined()) || (!has_sh && !colors_precomp.defined())) {
            throw std::invalid_argument(
                "Please provide exactly one of either SHs or precomputed colors!");
        }
        if (((scales.defined() || rotations.defined()) && cov3D_precomp.defined())
            || (!scales.defined() && !rotations.defined() && !cov3D_precomp.defined())) {
            throw std::invalid_argument(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!");
        }

        torch::Device device = torch::kCUDA;
        if (!shs_dc.defined()) {
            shs_dc = torch::empty({0}, device);
        }
        if (!shs_rest.defined()) {
            shs_rest = torch::empty({0}, device);
        }
        if (!colors_precomp.defined()) {
            colors_precomp = torch::empty({0}, device);
        }
        if (!scales.defined()) {
            scales = torch::empty({0}, device);
        }
        if (!rotations.defined()) {
            rotations = torch::empty({0}, device);
        }
        if (!cov3D_precomp.defined()) {
            cov3D_precomp = torch::empty({0}, device);
        }
        if (!theta.defined()) {
            theta = torch::empty({0}, device);
        }
        if (!rho.defined()) {
            rho = torch::empty({0}, device);
        }
        

        auto result = rasterize_gaussians_mono(means3D,
                                               means2D,
                                               shs_dc,
                                               shs_rest,
                                               colors_precomp,
                                               opacities,
                                               scales,
                                               rotations,
                                               cov3D_precomp,
                                               theta,
                                               rho,
                                               raster_settings_,
                                               fisher_means3D_diag,
                                               fisher_opacity_diag);

        return result;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        torch::Tensor shs_dc = torch::Tensor(),
        torch::Tensor shs_rest = torch::Tensor(),
        torch::Tensor colors_precomp = torch::Tensor(),
        torch::Tensor scales = torch::Tensor(),
        torch::Tensor rotations = torch::Tensor(),
        torch::Tensor cov3D_precomp = torch::Tensor(),
        torch::Tensor theta = torch::Tensor(),
        torch::Tensor rho = torch::Tensor(),
        torch::Tensor fisher_means3D_diag = torch::Tensor(),
        torch::Tensor fisher_opacity_diag = torch::Tensor())
    {
        auto result = forward_all(means3D,
                                  means2D,
                                  opacities,
                                  shs_dc,
                                  shs_rest,
                                  colors_precomp,
                                  scales,
                                  rotations,
                                  cov3D_precomp,
                                  theta,
                                  rho,
                                  fisher_means3D_diag,
                                  fisher_opacity_diag);
        return {result[0], result[1]};
    }

    private:
    MonoGaussianRasterizationSettings raster_settings_;
};

;

} 
