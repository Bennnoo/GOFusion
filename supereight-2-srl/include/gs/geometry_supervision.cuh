/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */



#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <torch/torch.h>

#include "gs/gaussian_utils.cuh"

namespace gs {

struct GeometryTarget {
    torch::Tensor depth;       
    torch::Tensor normal;      
    torch::Tensor depth_mask;  
    torch::Tensor normal_mask; 

    bool defined() const
    {
        return depth.defined() && normal.defined() && depth_mask.defined()
            && normal_mask.defined();
    }
};

struct RenderResult {
    torch::Tensor rgb;
    torch::Tensor depth_raw;
    torch::Tensor alpha;
    torch::Tensor normal_raw;
    torch::Tensor means2D;
    torch::Tensor visibility;
    torch::Tensor radii;

    bool hasGeometry() const
    {
        return depth_raw.defined() && alpha.defined() && normal_raw.defined();
    }
};

struct GeometryLossResult {
    torch::Tensor total;
    torch::Tensor depth;
    torch::Tensor normal;
};

inline torch::Tensor makeDepthCpuSnapshot(const float* depth,
                                          const int width,
                                          const int height,
                                          const float near_plane = 0.0f,
                                          const float far_plane =
                                              std::numeric_limits<float>::infinity())
{
    if (!depth || width <= 0 || height <= 0) {
        throw std::invalid_argument("Cannot snapshot an empty input depth image");
    }
    auto snapshot = torch::empty(
        {1, height, width},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU));
    std::memcpy(snapshot.data_ptr<float>(),
                depth,
                static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float));
    float* snapshot_data = snapshot.data_ptr<float>();
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        const float value = snapshot_data[i];
        if (!std::isfinite(value) || value <= near_plane || value >= far_plane) {
            snapshot_data[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return snapshot;
}

inline GeometryTarget makeGeometryTarget(const torch::Tensor& depth_cpu,
                                         const Camera& camera,
                                         const float near_plane,
                                         const float far_plane,
                                         const torch::Device device = torch::Device(torch::kCUDA))
{
    TORCH_CHECK(depth_cpu.defined(), "Geometry supervision requires input depth");
    TORCH_CHECK(depth_cpu.device().is_cpu(), "Geometry depth snapshot must be on CPU");
    TORCH_CHECK(depth_cpu.scalar_type() == torch::kFloat32,
                "Geometry depth snapshot must be float32 metres");
    TORCH_CHECK(depth_cpu.dim() == 3 && depth_cpu.size(0) == 1,
                "Geometry depth snapshot must have shape [1,H,W]");
    TORCH_CHECK(depth_cpu.size(1) == static_cast<int64_t>(camera.height)
                    && depth_cpu.size(2) == static_cast<int64_t>(camera.width),
                "Geometry supervision requires input depth already aligned to RGB");
    TORCH_CHECK(camera.fx > 0.0f && camera.fy > 0.0f,
                "Geometry supervision requires valid camera intrinsics");

    GeometryTarget target;

    target.depth = device.is_cpu()
        ? depth_cpu.contiguous()
        : depth_cpu.to(device, torch::kFloat32, true, true).contiguous();
    target.depth_mask = torch::isfinite(target.depth)
        & (target.depth > near_plane) & (target.depth < far_plane);

    const int64_t height = target.depth.size(1);
    const int64_t width = target.depth.size(2);
    auto options = target.depth.options();
    target.normal = torch::zeros({3, height, width}, options);
    target.normal_mask = torch::zeros(
        {1, height, width}, options.dtype(torch::kBool));
    if (height < 3 || width < 3) {
        return target;
    }

    using torch::indexing::Slice;
    const auto u = (torch::arange(width, options) + 0.5f - camera.cx) / camera.fx;
    const auto v = (torch::arange(height, options) + 0.5f - camera.cy) / camera.fy;
    const auto depth = target.depth.squeeze(0);
    const auto points = torch::stack(
        {depth * u.unsqueeze(0), depth * v.unsqueeze(1), depth}, 0);

    const auto dx = points.index({Slice(), Slice(1, height - 1), Slice(2, width)})
        - points.index({Slice(), Slice(1, height - 1), Slice(0, width - 2)});
    const auto dy = points.index({Slice(), Slice(2, height), Slice(1, width - 1)})
        - points.index({Slice(), Slice(0, height - 2), Slice(1, width - 1)});
    auto normal = torch::cross(dx, dy, 0);
    const auto norm = torch::sqrt(torch::sum(normal * normal, 0));

    const auto valid = target.depth_mask.squeeze(0);
    auto normal_valid = valid.index({Slice(1, height - 1), Slice(1, width - 1)})
        & valid.index({Slice(1, height - 1), Slice(0, width - 2)})
        & valid.index({Slice(1, height - 1), Slice(2, width)})
        & valid.index({Slice(0, height - 2), Slice(1, width - 1)})
        & valid.index({Slice(2, height), Slice(1, width - 1)})
        & torch::isfinite(norm) & (norm > 1.0e-12f);

    normal = normal / torch::clamp_min(norm, 1.0e-12f).unsqueeze(0);
    const auto center_points =
        points.index({Slice(), Slice(1, height - 1), Slice(1, width - 1)});
    const auto points_away_from_camera = torch::sum(normal * center_points, 0) > 0.0f;
    normal = torch::where(points_away_from_camera.unsqueeze(0), -normal, normal);
    normal = torch::where(normal_valid.unsqueeze(0), normal, torch::zeros_like(normal));

    target.normal.index_put_(
        {Slice(), Slice(1, height - 1), Slice(1, width - 1)}, normal);
    target.normal_mask.index_put_(
        {0, Slice(1, height - 1), Slice(1, width - 1)}, normal_valid);
    return target;
}

inline GeometryTarget geometryTargetToDevice(const GeometryTarget& source,
                                             const torch::Device device)
{
    TORCH_CHECK(source.defined(), "Cannot transfer an undefined geometry target");
    if (source.depth.device() == device) {
        return source;
    }
    GeometryTarget target;
    target.depth = source.depth.to(device, true);
    target.normal = source.normal.to(device, true);
    target.depth_mask = source.depth_mask.to(device, true);
    target.normal_mask = source.normal_mask.to(device, true);
    return target;
}

inline GeometryLossResult geometrySupervisionLoss(const RenderResult& rendered,
                                                  const GeometryTarget& target,
                                                  const float lambda_depth,
                                                  const float lambda_normal)
{
    TORCH_CHECK(rendered.rgb.defined(), "Geometry loss requires a rendered RGB image");
    TORCH_CHECK(rendered.hasGeometry(),
                "Geometry supervision is enabled but the selected rasterizer did not return D/A/N");
    TORCH_CHECK(target.defined(), "Geometry supervision is enabled without an input-depth target");
    TORCH_CHECK(rendered.depth_raw.sizes() == target.depth.sizes(),
                "Rendered and input depth dimensions differ");
    TORCH_CHECK(rendered.normal_raw.sizes() == target.normal.sizes(),
                "Rendered and input normal dimensions differ");

    auto zero = rendered.rgb.sum() * 0.0f;
    GeometryLossResult result{zero, zero, zero};

    if (lambda_depth > 0.0f) {
        const auto valid = target.depth_mask & torch::isfinite(rendered.depth_raw);
        const auto weights = valid.to(rendered.depth_raw.scalar_type());
        const auto safe_rendered = torch::where(
            valid, rendered.depth_raw, torch::zeros_like(rendered.depth_raw));
        const auto safe_target = torch::where(
            valid, target.depth, torch::zeros_like(target.depth));
        result.depth = torch::abs(safe_rendered - safe_target).sum()
            / torch::clamp_min(weights.sum(), 1.0f);
        result.total = result.total + lambda_depth * result.depth;
    }

    if (lambda_normal > 0.0f) {
        const auto valid = target.normal_mask
            & torch::isfinite(rendered.normal_raw).all(0, true)
            & torch::isfinite(rendered.alpha);
        const auto weights = valid.to(rendered.normal_raw.scalar_type());
        const auto safe_rendered = torch::where(
            valid.expand_as(rendered.normal_raw),
            rendered.normal_raw,
            torch::zeros_like(rendered.normal_raw));
        const auto safe_alpha = torch::where(
            valid, rendered.alpha.detach(), torch::zeros_like(rendered.alpha));
        const auto target_normal = safe_alpha * target.normal;
        const auto normal_error =
            torch::where(valid,
                         1.0f - torch::sum(safe_rendered * target_normal, 0, true),
                         torch::zeros_like(rendered.alpha));
        result.normal = normal_error.sum()
            / torch::clamp_min(weights.sum(), 1.0f);
        result.total = result.total + lambda_normal * result.normal;
    }
    return result;
}

} 
