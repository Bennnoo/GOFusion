// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#pragma once

#include "fastergs/buffer_utils.h"
#include "fastergs/helper_math.h"
#include "gs/raster_count_summary.cuh"
#include <functional>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<int, int, int, int, int> forward(
        std::function<char* (size_t)> resize_primitive_buffers,
        std::function<char* (size_t)> resize_tile_buffers,
        std::function<char* (size_t)> resize_instance_buffers,
        std::function<char* (size_t)> resize_bucket_buffers,
        std::function<char* (size_t)> resize_geometry_primitive_buffers,
        std::function<char* (size_t)> resize_geometry_bucket_buffers,
        std::function<uint32_t(uint32_t)> reserve_instance_capacity,
        std::function<uint32_t(uint32_t)> reserve_bucket_capacity,
        uint32_t initial_instance_capacity,
        uint32_t initial_bucket_capacity,
        gs::detail::RasterCountSummary* summary_device,
        std::function<const gs::detail::RasterCountSummary&()> read_summary,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        float* image,
        float* depth_image,
        float* alpha_image,
        float* normal_image,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool render_geometry);

}
