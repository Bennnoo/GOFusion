// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#include "fastergs/backward.h"
#include "fastergs/kernels_backward.cuh"
#include "fastergs/buffer_utils.h"
#include "fastergs/rasterization_config.h"
#include "fastergs/utils.h"
#include "fastergs/helper_math.h"
#include "gs/nvtx_utils.hpp"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

void faster_gs::rasterization::backward(
    const float* grad_image,
    const float* image,
    const float* grad_depth,
    const float* grad_alpha,
    const float* grad_normal,
    const float* depth_image,
    const float* normal_image,
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    char* bucket_buffers_blob,
    char* geometry_primitive_buffers_blob,
    char* geometry_bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales,
    float4* grad_rotations,
    float* grad_opacities,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float4* grad_planes,
    float* densification_info,
    const int n_primitives,
    const int n_instances,
    const int n_buckets,
    const int instance_capacity,
    const int bucket_capacity,
    const int instance_primitive_indices_selector,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const bool proper_antialiasing)
{
    GOFUSION_NVTX_RANGE("FasterGS/kernels/backward");
    // No contributing primitives: the forward image is background-only and
    // backward_wrapper has already zero-initialized all output gradients.
    // CUDA rejects a zero-block launch (n_buckets == 0).
    if (n_primitives == 0 || n_buckets == 0) {
        return;
    }
    const int device_index = c10::cuda::current_device();
    c10::cuda::CUDAGuard device_guard(device_index);
    const auto stream_object =
        c10::cuda::getCurrentCUDAStream(device_index);
    const cudaStream_t stream = stream_object.stream();
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = tile_key_end_bit(static_cast<uint>(n_tiles));

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    BucketBuffers bucket_buffers =
        BucketBuffers::from_blob(bucket_buffers_blob, bucket_capacity);
    const bool render_geometry = geometry_primitive_buffers_blob != nullptr;
    float4* primitive_planes = nullptr;
    float4* bucket_geometry = nullptr;
    if (render_geometry) {
        primitive_planes = GeometryPrimitiveBuffers::from_blob(
            geometry_primitive_buffers_blob, n_primitives).planes;
        bucket_geometry = GeometryBucketBuffers::from_blob(
            geometry_bucket_buffers_blob, bucket_capacity).depth_normal;
    }

    gs::profiling::NvtxRange blend_backward_range("FasterGS/kernels/blend_backward");
    auto dispatch_rasterize_backward = [&](const uint* instance_primitive_indices) {
#define FASTER_BLEND_BACKWARD_ARGS                                                \
			tile_buffers.instance_ranges,                                             \
			tile_buffers.buckets_offset,                                              \
			instance_primitive_indices,                                               \
			primitive_buffers.mean2d,                                                 \
			primitive_buffers.conic_opacity,                                          \
			primitive_buffers.color,                                                  \
			primitive_planes,                                                         \
			bg_color,                                                                 \
			grad_image,                                                               \
			image,                                                                    \
			grad_depth,                                                               \
			grad_alpha,                                                               \
			grad_normal,                                                              \
			depth_image,                                                              \
			normal_image,                                                             \
			tile_buffers.final_transmittances,                                        \
			tile_buffers.max_n_processed,                                             \
			tile_buffers.n_processed,                                                 \
			bucket_buffers.tile_index,                                                \
			bucket_buffers.color_transmittance,                                       \
			bucket_geometry,                                                          \
			grad_mean2d_helper,                                                       \
			grad_conic_helper,                                                        \
			grad_opacities,                                                           \
			grad_sh_coefficients_0,                                                   \
			grad_planes,                                                              \
						n_primitives,                                                             \
			width,                                                                    \
			height,                                                                   \
			grid.x,                                                                  \
			focal_x,                                                                 \
			focal_y,                                                                 \
			center_x,                                                                \
			center_y,                                                                \
			proper_antialiasing
        if (render_geometry) {
            kernels::backward::blend_backward_cu<true><<<n_buckets, 32, 0, stream>>>(
                FASTER_BLEND_BACKWARD_ARGS);
        }
        else {
            kernels::backward::blend_backward_cu<false><<<n_buckets, 32, 0, stream>>>(
                FASTER_BLEND_BACKWARD_ARGS);
        }
#undef FASTER_BLEND_BACKWARD_ARGS
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (end_bit <= 16) {
        auto instance_buffers =
            InstanceBuffers<ushort>::from_blob(
                instance_buffers_blob, instance_capacity, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_rasterize_backward(instance_buffers.primitive_indices.Current());
    }
    else {
        auto instance_buffers =
            InstanceBuffers<uint>::from_blob(
                instance_buffers_blob, instance_capacity, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_rasterize_backward(instance_buffers.primitive_indices.Current());
    }
    blend_backward_range.stop();

    const bool need_preprocess = grad_mean2d_helper != nullptr || grad_conic_helper != nullptr
        || grad_means != nullptr || grad_scales != nullptr || grad_rotations != nullptr
        || grad_opacities != nullptr || grad_sh_coefficients_0 != nullptr
        || grad_sh_coefficients_rest != nullptr || densification_info != nullptr;
    if (need_preprocess) {
        gs::profiling::NvtxRange preprocess_backward_range(
            "FasterGS/kernels/preprocess_backward");
        kernels::backward::preprocess_backward_cu<<<
            div_round_up(n_primitives, config::block_size_preprocess_backward),
            config::block_size_preprocess_backward,
            0,
            stream>>>(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            w2c,
            cam_position,
            primitive_buffers.n_touched_tiles,
            grad_mean2d_helper,
            grad_conic_helper,
            grad_planes,
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            densification_info,
            n_primitives,
            active_sh_bases,
            total_sh_bases,
            static_cast<float>(width),
            static_cast<float>(height),
            focal_x,
            focal_y,
            center_x,
            center_y,
            proper_antialiasing
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        preprocess_backward_range.stop();
    }

}
