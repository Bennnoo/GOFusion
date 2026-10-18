// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#pragma once

#include "fastergs/rasterization_config.h"
#include "fastergs/kernel_utils.cuh"
#include "fastergs/sh_utils.cuh"
#include "fastergs/buffer_utils.h"
#include "fastergs/helper_math.h"
#include "fastergs/utils.h"
#include "gs/raster_count_summary.cuh"
#include "gs/surfel_geometry.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef FASTERGS_FORWARD_KERNEL_NAMESPACE
#define FASTERGS_FORWARD_KERNEL_NAMESPACE forward
#define FASTERGS_UNDEFINE_FORWARD_KERNEL_NAMESPACE
#endif

namespace faster_gs::rasterization::kernels::FASTERGS_FORWARD_KERNEL_NAMESPACE {

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        float4* __restrict__ primitive_planes,
        uint* __restrict__ n_visible_primitives,
        uint* __restrict__ n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing)
    {
        if (n_primitives == 0) return;

        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();

        bool active = true;
        uint primitive_idx = thread_idx;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (active) primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        float3 mean3d = means[primitive_idx];
        if (!finite_value(mean3d)) {
            active = false;
            mean3d = make_float3(0.0f);
        }

        // z culling
        const float4 w2c_r3 = w2c[2];
        float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (!finite_value(depth)) {
            active = false;
            depth = 1.0f;
        }
        if (depth < near_plane || depth > far_plane) active = false;

        // early exit if whole warp is inactive
        if (warp.ballot(active) == 0) return;

        // load opacity
        float raw_opacity = opacities[primitive_idx];
        if (!finite_value(raw_opacity)) {
            active = false;
            raw_opacity = 0.0f;
        }
        float opacity = sigmoid(raw_opacity);
        if (!finite_value(opacity)) {
            active = false;
            opacity = 0.0f;
        }
        if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;

        // compute 3d covariance from scale and rotation
        float3 raw_scale = scales[primitive_idx];
        if (!finite_value(raw_scale)) {
            active = false;
            raw_scale = make_float3(0.0f);
        }
        float3 variance = expf(2.0f * raw_scale);
        if (!finite_value(variance)) {
            active = false;
            variance = make_float3(1.0f);
        }
        float4 raw_rotation = rotations[primitive_idx];
        if (!finite_value(raw_rotation)) {
            active = false;
            raw_rotation = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        }
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
        if (!finite_value(quaternion_norm_sq) || quaternion_norm_sq < 1e-8f) active = false;
        const mat3x3 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        const mat3x3_triu cov3d {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };

        // compute 2d mean in normalized image coordinates
        const float4 w2c_r1 = w2c[0];
        float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;
        if (!finite_value(x) || !finite_value(y)) {
            active = false;
            x = 0.0f;
            y = 0.0f;
        }

        // ewa splatting
        const float clip_left = (-0.15f * width - center_x) / focal_x;
        const float clip_right = (1.15f * width - center_x) / focal_x;
        const float clip_top = (-0.15f * height - center_y) / focal_y;
        const float clip_bottom = (1.15f * height - center_y) / focal_y;
        const float x_clipped = clamp(x, clip_left, clip_right);
        const float y_clipped = clamp(y, clip_top, clip_bottom);
        const float j11 = focal_x / depth;
        const float j13 = -j11 * x_clipped;
        const float j22 = focal_y / depth;
        const float j23 = -j22 * y_clipped;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z
        );
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z
        );
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
        );
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2)
        );
        if (!finite_value(cov2d)) {
            active = false;
            cov2d = make_float3(1.0f, 0.0f, 1.0f);
        }
        float determinant_raw = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (!finite_value(determinant_raw)) {
            active = false;
            determinant_raw = 1.0f;
        }
        const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
        cov2d.x += kernel_size;
        cov2d.z += kernel_size;
        const float determinant_raw_dilated = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        const bool determinant_valid = finite_value(determinant_raw_dilated)
            && determinant_raw_dilated >= config::min_cov2d_determinant;
        if (!determinant_valid) active = false;
        const float determinant = determinant_valid ? determinant_raw_dilated : 1.0f;
        float3 conic = make_float3(
            cov2d.z / determinant, -cov2d.y / determinant, cov2d.x / determinant
        );
        if (!finite_value(conic)) {
            active = false;
            conic = make_float3(1.0f, 0.0f, 1.0f);
        }
        if (proper_antialiasing) {
            opacity *= sqrtf(fmaxf(determinant_raw / determinant, 0.0f));
            if (!finite_value(opacity)) {
                active = false;
                opacity = 0.0f;
            }
            if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;
        }

        // 2d mean in screen space
        float2 mean2d = make_float2(
            x * focal_x + center_x,
            y * focal_y + center_y
        );
        if (!finite_value(mean2d)) {
            active = false;
            mean2d = make_float2(0.0f);
        }

        // compute bounds
        float power_threshold = config::original_opacity_interpretation ? logf(opacity * config::min_alpha_threshold_rcp) : config::max_power_threshold;
        if (!finite_value(power_threshold) || power_threshold < 0.0f) {
            active = false;
            power_threshold = 0.0f;
        }
        const float cutoff_factor = 2.0f * power_threshold;
        float extent_x = fmaxf(sqrtf(cov2d.x * cutoff_factor) - 0.5f, 0.0f);
        float extent_y = fmaxf(sqrtf(cov2d.z * cutoff_factor) - 0.5f, 0.0f);
        if (!finite_value(extent_x) || !finite_value(extent_y)) {
            active = false;
            extent_x = 0.0f;
            extent_y = 0.0f;
        }
        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
        );
        const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles_max == 0) active = false;

        // early exit if whole warp is inactive
        if (warp.ballot(active) == 0) return;

        // compute exact number of tiles the primitive overlaps
        const uint n_touched_tiles = compute_exact_n_touched_tiles(
            mean2d, conic, screen_bounds,
            power_threshold, n_touched_tiles_max, active
        );

         // cooperative threads no longer needed
        if (n_touched_tiles == 0 || !active) return;

        const float3 color = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_sh_bases
        );
        if (!finite_value(color)) return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);
        primitive_color[primitive_idx] = color;
        if (primitive_planes != nullptr) {
            const float3 normal_world = gs::surfel::quaternionNormal(raw_rotation);
            float3 normal_camera = make_float3(
                w2c_r1.x * normal_world.x + w2c_r1.y * normal_world.y
                    + w2c_r1.z * normal_world.z,
                w2c_r2.x * normal_world.x + w2c_r2.y * normal_world.y
                    + w2c_r2.z * normal_world.z,
                w2c_r3.x * normal_world.x + w2c_r3.y * normal_world.y
                    + w2c_r3.z * normal_world.z);
            const float3 center_camera = make_float3(x * depth, y * depth, depth);
            if (dot(normal_camera, center_camera) > 0.0f) normal_camera = -normal_camera;
            primitive_planes[primitive_idx] = make_float4(
                normal_camera, dot(normal_camera, center_camera));
        }

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void apply_depth_ordering_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_n_touched_tiles,
        uint* __restrict__ primitive_offset,
        const uint n_visible_primitives)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    __global__ void apply_depth_ordering_capacity_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_n_touched_tiles,
        uint* __restrict__ primitive_offset,
        const uint* __restrict__ n_visible_primitives,
        const uint primitive_capacity)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= primitive_capacity) return;
        const uint n_visible = min(*n_visible_primitives, primitive_capacity);
        primitive_offset[idx] = idx < n_visible
            ? primitive_n_touched_tiles[primitive_indices_sorted[idx]]
            : 0u;
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_n_touched_tiles,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        uint* __restrict__ validation_error,
        const uint grid_width,
        const uint n_visible_primitives,
        const uint n_instances)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible_primitives) {
            active = false;
            original_idx = n_visible_primitives - 1;
        }

        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;
        const float2 mean2d = primitive_mean2d[primitive_idx];
        const float2 mean2d_shifted = mean2d - 0.5f;
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const float3 conic = make_float3(conic_opacity);
        const float opacity = conic_opacity.w;
        const float power_threshold = config::original_opacity_interpretation ? logf(opacity * config::min_alpha_threshold_rcp) : config::max_power_threshold;

        const uint write_begin = primitive_offsets[original_idx];
        const uint expected_count = primitive_n_touched_tiles[primitive_idx];
        const bool write_range_valid = write_begin <= n_instances
            && expected_count <= n_instances - write_begin;
        const uint write_end = write_range_valid ? write_begin + expected_count : n_instances;
        if (active && !write_range_valid) atomicOr(validation_error, 16u);
        uint current_write_offset = write_begin;

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                if (current_write_offset < write_end) {
                    const uint tile_idx = tile_y * grid_width + tile_x;
                    instance_keys[current_write_offset] = static_cast<KeyT>(tile_idx);
                    instance_primitive_indices[current_write_offset] = primitive_idx;
                }
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (active && !compute_cooperatively && current_write_offset != write_end)
            atomicOr(validation_error, 8u);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);
            const uint write_end_coop = warp.shfl(write_end, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];
            const float2 mean2d_shifted_coop = collected_mean2d_shifted[read_offset_shared];
            const float4 conic_power_threshold_coop = collected_conic_power_threshold[read_offset_shared];

            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint instance_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);
            const float3 conic_coop = make_float3(conic_power_threshold_coop);
            const float power_threshold_coop = conic_power_threshold_coop.w;

            const uint remaining_instance_count = instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const bool write = instance_idx < instance_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask);
                    if (write_offset < write_end_coop) {
                        const uint tile_idx = tile_y * grid_width + tile_x;
                        instance_keys[write_offset] = static_cast<KeyT>(tile_idx);
                        instance_primitive_indices[write_offset] = primitive_idx_coop;
                    }
                }
                const uint n_written = __popc(write_ballot);
                current_write_offset_coop += n_written;
            }
            if (lane_idx == current_lane && current_write_offset_coop != write_end_coop)
                atomicOr(validation_error, 8u);
            warp.sync();
        }
    }

    template <typename KeyT>
    __global__ void create_instances_capacity_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_n_touched_tiles,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        uint* __restrict__ validation_error,
        const uint grid_width,
        const uint* __restrict__ n_visible_primitives,
        const uint primitive_capacity,
        const uint instance_capacity)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1u << lane_idx) - 1u;
        const uint n_visible = min(*n_visible_primitives, primitive_capacity);

        uint original_idx = thread_idx;
        bool active = original_idx < n_visible;
        if (!active) original_idx = n_visible == 0 ? 0 : n_visible - 1;
        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];
        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count =
            static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;
        const float2 mean2d_shifted = primitive_mean2d[primitive_idx] - 0.5f;
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const float3 conic = make_float3(conic_opacity);
        const float power_threshold = config::original_opacity_interpretation
            ? logf(conic_opacity.w * config::min_alpha_threshold_rcp)
            : config::max_power_threshold;

        const uint write_begin = primitive_offsets[original_idx];
        const uint expected_count = primitive_n_touched_tiles[primitive_idx];
        const uint expected_end = write_begin + expected_count;
        if (active && expected_end < write_begin) {
            atomicOr(validation_error, 16u);
            active = false;
        }
        if (active && expected_end > instance_capacity) {
            atomicOr(validation_error, gs::detail::kRasterOverflowInstances);
        }
        uint current_write_offset = write_begin;

        for (uint instance_idx = 0;
             active && instance_idx < instance_count
                 && instance_idx < config::n_sequential_threshold;
             ++instance_idx) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            if (will_primitive_contribute(
                    mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                if (current_write_offset < instance_capacity) {
                    instance_keys[current_write_offset] =
                        static_cast<KeyT>(tile_y * grid_width + tile_x);
                    instance_primitive_indices[current_write_offset] = primitive_idx;
                }
                ++current_write_offset;
            }
        }

        const bool compute_cooperatively =
            active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (active && !compute_cooperatively && current_write_offset != expected_end)
            atomicOr(validation_error, 8u);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; ++n) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);
            const uint expected_end_coop = warp.shfl(expected_end, current_lane);
            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];
            const float2 mean2d_shifted_coop =
                collected_mean2d_shifted[read_offset_shared];
            const float4 conic_power_threshold_coop =
                collected_conic_power_threshold[read_offset_shared];
            const uint screen_bounds_width_coop =
                static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint instance_count_coop = screen_bounds_width_coop
                * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);
            const float3 conic_coop = make_float3(conic_power_threshold_coop);
            const float power_threshold_coop = conic_power_threshold_coop.w;
            const uint remaining_instance_count =
                instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; ++i) {
                const uint instance_idx =
                    i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x =
                    screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint tile_y =
                    screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const bool write = instance_idx < instance_count_coop
                    && will_primitive_contribute(mean2d_shifted_coop,
                                                 conic_coop,
                                                 tile_x,
                                                 tile_y,
                                                 power_threshold_coop);
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop
                        + __popc(write_ballot & previous_lanes_mask);
                    if (write_offset < instance_capacity) {
                        instance_keys[write_offset] =
                            static_cast<KeyT>(tile_y * grid_width + tile_x);
                        instance_primitive_indices[write_offset] = primitive_idx_coop;
                    }
                }
                current_write_offset_coop += __popc(write_ballot);
            }
            if (lane_idx == current_lane
                && current_write_offset_coop != expected_end_coop)
                atomicOr(validation_error, 8u);
            warp.sync();
        }
    }

    template <typename KeyT>
    __global__ void validate_sorted_instance_keys_capacity_cu(
        const KeyT* __restrict__ instance_keys,
        const uint* __restrict__ n_instances,
        const uint instance_capacity,
        const uint n_tiles,
        uint* __restrict__ validation_error)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint effective = min(*n_instances, instance_capacity);
        if (idx >= effective) return;
        const uint tile = static_cast<uint>(instance_keys[idx]);
        if (tile >= n_tiles) {
            atomicOr(validation_error, 1u);
            return;
        }
        if (idx > 0 && instance_keys[idx - 1] > instance_keys[idx])
            atomicOr(validation_error, 4u);
    }

    template <typename KeyT>
    __global__ void build_instance_ranges_capacity_cu(
        const KeyT* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ n_instances,
        const uint instance_capacity,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint effective = min(*n_instances, instance_capacity);
        uint lower_first = 0;
        uint lower_count = effective;
        while (lower_count > 0) {
            const uint step = lower_count / 2;
            const uint probe = lower_first + step;
            if (static_cast<uint>(instance_keys[probe]) < tile_idx) {
                lower_first = probe + 1;
                lower_count -= step + 1;
            }
            else lower_count = step;
        }
        uint upper_first = lower_first;
        uint upper_count = effective - upper_first;
        while (upper_count > 0) {
            const uint step = upper_count / 2;
            const uint probe = upper_first + step;
            if (static_cast<uint>(instance_keys[probe]) <= tile_idx) {
                upper_first = probe + 1;
                upper_count -= step + 1;
            }
            else upper_count = step;
        }
        tile_instance_ranges[tile_idx] = make_uint2(lower_first, upper_first);
    }

    template <typename KeyT>
    __global__ void validate_sorted_instance_keys_cu(
        const KeyT* __restrict__ instance_keys,
        const uint n_instances,
        const uint n_tiles,
        uint* __restrict__ validation_error)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (static_cast<uint>(instance_tile_idx) >= n_tiles) {
            atomicOr(validation_error, 1u);
            return;
        }
        if (instance_idx > 0) {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (previous_instance_tile_idx > instance_tile_idx)
                atomicOr(validation_error, 4u);
        }
    }

    template <typename KeyT>
    __global__ void build_instance_ranges_cu(
        const KeyT* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;

        uint lower_first = 0;
        uint lower_count = n_instances;
        while (lower_count > 0) {
            const uint step = lower_count / 2;
            const uint probe = lower_first + step;
            if (static_cast<uint>(instance_keys[probe]) < tile_idx) {
                lower_first = probe + 1;
                lower_count -= step + 1;
            }
            else lower_count = step;
        }

        uint upper_first = lower_first;
        uint upper_count = n_instances - upper_first;
        while (upper_count > 0) {
            const uint step = upper_count / 2;
            const uint probe = upper_first + step;
            if (static_cast<uint>(instance_keys[probe]) <= tile_idx) {
                upper_first = probe + 1;
                upper_count -= step + 1;
            }
            else upper_count = step;
        }

        tile_instance_ranges[tile_idx] = make_uint2(lower_first, upper_first);
    }

    __global__ void extract_bucket_counts(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_n_buckets,
        const uint n_tiles,
        const uint n_instances,
        uint* __restrict__ validation_error)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        if (instance_range.x > instance_range.y || instance_range.y > n_instances) {
            tile_n_buckets[tile_idx] = 0;
            atomicOr(validation_error, 2u);
            return;
        }
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
        tile_n_buckets[tile_idx] = n_buckets;
    }

    template<bool RENDER_GEOMETRY>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_buckets_offset,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float4* __restrict__ primitive_planes,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ tile_final_transmittances,
        uint* __restrict__ tile_max_n_processed,
        uint* __restrict__ tile_n_processed,
        uint* __restrict__ bucket_tile_index,
        float4* __restrict__ bucket_color_transmittance,
        float4* __restrict__ bucket_geometry,
        float* __restrict__ depth_image,
        float* __restrict__ alpha_image,
        float* __restrict__ normal_image,
        const uint width,
        const uint height,
        const uint grid_width,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup tile info
        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;
        // setup bucket to tile mapping
        const int n_buckets = div_round_up(n_points_total, 32);
        uint bucket_offset = (tile_idx == 0) ? 0 : tile_buckets_offset[tile_idx - 1];
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets) bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }
        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        __shared__ float4 collected_plane[
            RENDER_GEOMETRY ? config::block_size_blend : 1];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float depth_pixel = 0.0f;
        float3 normal_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_processed = 0;
        uint n_processed_and_used = 0;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_color[thread_rank] = color;
                if constexpr (RENDER_GEOMETRY) {
                    collected_plane[thread_rank] = primitive_planes[primitive_idx];
                }
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // store current color and transmittance every 32 Gaussians
                if (j % 32 == 0) {
                    const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + thread_rank] = current_color_transmittance;
                    if constexpr (RENDER_GEOMETRY) {
                        bucket_geometry[bucket_offset * config::block_size_blend + thread_rank] =
                            make_float4(depth_pixel,
                                        normal_pixel.x,
                                        normal_pixel.y,
                                        normal_pixel.z);
                    }
                    bucket_offset++;
                }

                // track the number of processed Gaussians
                n_processed++;

                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
                const float alpha = opacity * gaussian;
                if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;

                // blend fragment into pixel color
                const float blending_weight = transmittance * alpha;
                color_pixel += blending_weight * collected_color[j];
                if constexpr (RENDER_GEOMETRY) {
                    const float4 plane = collected_plane[j];
                    const float3 ray = make_float3(
                        (pixel.x - center_x) / focal_x,
                        (pixel.y - center_y) / focal_y,
                        1.0f);
                    depth_pixel += blending_weight * gs::surfel::rayPlaneDepth(plane, ray);
                    normal_pixel += blending_weight * make_float3(plane);
                }

                // update transmittance
                transmittance *= 1.0f - alpha;

                // update the number of used Gaussians
                n_processed_and_used = n_processed;

                // early stopping
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
        if (inside) {
            // apply background color
            color_pixel += transmittance * bg_color[0];
            // store results
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint n_pixels = width * height;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            tile_final_transmittances[pixel_idx] = transmittance;
            tile_n_processed[pixel_idx] = n_processed_and_used;
            if constexpr (RENDER_GEOMETRY) {
                depth_image[pixel_idx] = depth_pixel;
                alpha_image[pixel_idx] = 1.0f - transmittance;
                normal_image[pixel_idx] = normal_pixel.x;
                normal_image[n_pixels + pixel_idx] = normal_pixel.y;
                normal_image[2 * n_pixels + pixel_idx] = normal_pixel.z;
            }
        }
        // max reduce the number of processed Gaussians per tile
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
#if defined(CUB_MAJOR_VERSION) && CUB_MAJOR_VERSION >= 3
        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, ::cuda::maximum<>{});
#else
        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());
#endif
        if (thread_rank == 0) tile_max_n_processed[tile_idx] = n_processed_and_used;
    }

}

#ifdef FASTERGS_UNDEFINE_FORWARD_KERNEL_NAMESPACE
#undef FASTERGS_UNDEFINE_FORWARD_KERNEL_NAMESPACE
#undef FASTERGS_FORWARD_KERNEL_NAMESPACE
#endif
