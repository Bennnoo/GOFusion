// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)

#include "fastergs/forward.h"

#include "fastergs/buffer_utils.h"
#include "fastergs/helper_math.h"
#include "fastergs/kernels_forward.cuh"
#include "fastergs/rasterization_config.h"
#include "fastergs/utils.h"
#include "gs/nvtx_utils.hpp"
#include "gs/raster_count_summary.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

using gs::detail::RasterCountSummary;

__global__ void packFasterRasterSummary(
    RasterCountSummary* summary,
    const uint* n_visible,
    const uint* n_instances,
    const uint* bucket_offsets,
    const uint* validation_error,
    const uint n_primitives,
    const uint n_tiles,
    const uint instance_capacity,
    const uint bucket_capacity)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RasterCountSummary result{};
    result.n_visible = *n_visible;
    result.n_instances = *n_instances;
    result.n_buckets = n_tiles == 0 ? 0u : bucket_offsets[n_tiles - 1];
    result.error_flags = *validation_error;
    result.required_instance_capacity = result.n_instances;
    result.required_bucket_capacity = result.n_buckets;
    result.detail0 = instance_capacity;
    result.detail1 = bucket_capacity;

    const uint64_t max_instances =
        static_cast<uint64_t>(result.n_visible) * static_cast<uint64_t>(n_tiles);
    if (result.n_visible > n_primitives
        || static_cast<uint64_t>(result.n_instances) > max_instances
        || result.n_buckets > result.n_instances) {
        result.error_flags |= gs::detail::kRasterInvalidCounts;
    }
    if (result.n_instances > instance_capacity)
        result.error_flags |= gs::detail::kRasterOverflowInstances;
    if (result.n_buckets > bucket_capacity)
        result.error_flags |= gs::detail::kRasterOverflowBuckets;
    *summary = result;
}

__global__ void packFasterBootstrapSummary(
    RasterCountSummary* summary,
    const uint* n_visible,
    const uint* n_instances,
    const uint n_primitives,
    const uint n_tiles)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RasterCountSummary result{};
    result.n_visible = *n_visible;
    result.n_instances = *n_instances;
    result.required_instance_capacity = result.n_instances;
    const uint64_t maximum =
        static_cast<uint64_t>(result.n_visible) * static_cast<uint64_t>(n_tiles);
    if (result.n_visible > n_primitives
        || static_cast<uint64_t>(result.n_instances) > maximum)
        result.error_flags |= gs::detail::kRasterInvalidCounts;
    *summary = result;
}

uint32_t initialInstanceCapacity(const int n_primitives)
{
    (void) n_primitives;
    return 4096u;
}

uint32_t initialBucketCapacity(const uint32_t instance_capacity, const int n_tiles)
{
    const uint64_t estimate = std::max<uint64_t>(
        1024u,
        static_cast<uint64_t>(instance_capacity) / 16u
            + static_cast<uint64_t>(std::max(n_tiles, 0)));
    return static_cast<uint32_t>(
        std::min<uint64_t>(estimate, std::numeric_limits<uint32_t>::max()));
}

void throwFasterValidationError(const uint32_t flags)
{
    if ((flags & gs::detail::kRasterInvalidCounts) != 0)
        throw std::runtime_error("FasterGS rasterizer produced invalid packed counts");
    if ((flags & 16u) != 0)
        throw std::runtime_error("FasterGS rasterizer produced an invalid instance write range");
    if ((flags & 8u) != 0)
        throw std::runtime_error("FasterGS rasterizer instance count/write mismatch");
    if ((flags & ~(gs::detail::kRasterOverflowInstances
                   | gs::detail::kRasterOverflowBuckets)) != 0) {
        throw std::runtime_error(
            "FasterGS rasterizer detected invalid/unsorted tile keys or tile ranges");
    }
}

template<typename KeyT>
std::tuple<int, int, int, int, int> rasterizeCapacityAware(
	std::function<char*(size_t)>& resize_instance_buffers,
	std::function<char*(size_t)>& resize_bucket_buffers,
	std::function<char*(size_t)>& resize_geometry_bucket_buffers,
    std::function<uint32_t(uint32_t)>& reserve_instance_capacity,
    std::function<uint32_t(uint32_t)>& reserve_bucket_capacity,
    uint32_t instance_capacity,
    uint32_t bucket_capacity,
    RasterCountSummary* summary_device,
    std::function<const RasterCountSummary&()>& read_summary,
    faster_gs::rasterization::PrimitiveBuffers& primitive_buffers,
	faster_gs::rasterization::TileBuffers& tile_buffers,
	const float4* primitive_planes,
    const dim3& grid,
    const dim3& block,
    const float3* bg_color,
	float* image,
	float* depth_image,
	float* alpha_image,
	float* normal_image,
    const int n_primitives,
    const int n_tiles,
    const int end_bit,
    const int width,
	const int height,
	const float focal_x,
	const float focal_y,
	const float center_x,
	const float center_y,
	const bool render_geometry,
	const cudaStream_t stream)
{
    using namespace faster_gs::rasterization;
    instance_capacity = reserve_instance_capacity(std::max(
        instance_capacity, initialInstanceCapacity(n_primitives)));
    bucket_capacity = reserve_bucket_capacity(std::max(
        bucket_capacity, initialBucketCapacity(instance_capacity, n_tiles)));

    for (int attempt = 0; attempt < 2; ++attempt) {
        GOFUSION_NVTX_RANGE("FasterGS/kernels/capacity_attempt");
        cudaMemsetAsync(tile_buffers.validation_error, 0, sizeof(uint), stream);

        char* instance_blob = resize_instance_buffers(
            required<InstanceBuffers<KeyT>>(instance_capacity, end_bit));
        auto instance_buffers =
            InstanceBuffers<KeyT>::from_blob(instance_blob, instance_capacity, end_bit);
        if (instance_capacity > 0) {
            cudaMemsetAsync(instance_buffers.keys.Current(),
                            0xff,
                            static_cast<size_t>(instance_capacity) * sizeof(KeyT),
                            stream);
            cudaMemsetAsync(instance_buffers.primitive_indices.Current(),
                            0xff,
                            static_cast<size_t>(instance_capacity) * sizeof(uint),
                            stream);
        }

        if (n_primitives > 0) {
            kernels::forward::create_instances_capacity_cu<KeyT><<<
                div_round_up(n_primitives, config::block_size_create_instances),
                config::block_size_create_instances,
                0,
                stream>>>(primitive_buffers.primitive_indices.Current(),
                          primitive_buffers.offset,
                          primitive_buffers.n_touched_tiles,
                          primitive_buffers.screen_bounds,
                          primitive_buffers.mean2d,
                          primitive_buffers.conic_opacity,
                          instance_buffers.keys.Current(),
                          instance_buffers.primitive_indices.Current(),
                          tile_buffers.validation_error,
                          grid.x,
                          primitive_buffers.n_visible_primitives,
                          static_cast<uint>(n_primitives),
                          instance_capacity);
        }
        CHECK_CUDA(config::debug, "create_instances_capacity")

        cub::DeviceRadixSort::SortPairs(instance_buffers.cub_workspace,
                                        instance_buffers.cub_workspace_size,
                                        instance_buffers.keys,
                                        instance_buffers.primitive_indices,
                                        instance_capacity,
                                        0,
                                        end_bit,
                                        stream);
        CHECK_CUDA(config::debug, "SortPairs(tile capacity)")
        const int selector = instance_buffers.primitive_indices.selector;

        if (instance_capacity > 0) {
            kernels::forward::validate_sorted_instance_keys_capacity_cu<KeyT><<<
                div_round_up(instance_capacity,
                             static_cast<uint>(config::block_size_extract_instance_ranges)),
                config::block_size_extract_instance_ranges,
                0,
                stream>>>(instance_buffers.keys.Current(),
                          primitive_buffers.n_instances,
                          instance_capacity,
                          n_tiles,
                          tile_buffers.validation_error);
        }
        kernels::forward::build_instance_ranges_capacity_cu<KeyT><<<
            div_round_up(n_tiles, config::block_size_extract_instance_ranges),
            config::block_size_extract_instance_ranges,
            0,
            stream>>>(instance_buffers.keys.Current(),
                      tile_buffers.instance_ranges,
                      primitive_buffers.n_instances,
                      instance_capacity,
                      n_tiles);
        kernels::forward::extract_bucket_counts<<<
            div_round_up(n_tiles, config::block_size_extract_bucket_counts),
            config::block_size_extract_bucket_counts,
            0,
            stream>>>(tile_buffers.instance_ranges,
                      tile_buffers.n_buckets,
                      n_tiles,
                      instance_capacity,
                      tile_buffers.validation_error);
        cub::DeviceScan::InclusiveSum(tile_buffers.cub_workspace,
                                      tile_buffers.cub_workspace_size,
                                      tile_buffers.n_buckets,
                                      tile_buffers.buckets_offset,
                                      n_tiles,
                                      stream);
        CHECK_CUDA(config::debug, "bucket scan")

        packFasterRasterSummary<<<1, 1, 0, stream>>>(summary_device,
                                                     primitive_buffers.n_visible_primitives,
                                                     primitive_buffers.n_instances,
                                                     tile_buffers.buckets_offset,
                                                     tile_buffers.validation_error,
                                                     n_primitives,
                                                     n_tiles,
                                                     instance_capacity,
                                                     bucket_capacity);
        CHECK_CUDA(config::debug, "pack raster summary")
        const RasterCountSummary summary = read_summary();
        auto& stats = gs::detail::rasterDependencyStats();
        gs::detail::updateAtomicMax(stats.instance_high_water, summary.n_instances);
        gs::detail::updateAtomicMax(stats.bucket_high_water, summary.n_buckets);
        gs::detail::maybeReportRasterDependencyStats(
            "FasterGS", summary, instance_capacity, bucket_capacity);

        const uint32_t overflow = summary.error_flags
            & (gs::detail::kRasterOverflowInstances
               | gs::detail::kRasterOverflowBuckets);
        if (overflow != 0) {
            if (attempt != 0) {
                throw std::runtime_error(
                    "FasterGS raster capacity overflow persisted after one retry: "
                    "instances=" + std::to_string(summary.n_instances)
                    + "/" + std::to_string(instance_capacity)
                    + ", buckets=" + std::to_string(summary.n_buckets)
                    + "/" + std::to_string(bucket_capacity));
            }
            stats.raster_retries.fetch_add(1, std::memory_order_relaxed);
            const uint32_t required_instances = std::max(
                summary.required_instance_capacity, instance_capacity);
            const uint64_t bucket_hint = static_cast<uint64_t>(required_instances) / 16u
                + static_cast<uint64_t>(n_tiles);
            const uint32_t required_buckets = static_cast<uint32_t>(
                std::min<uint64_t>(
                    std::max<uint64_t>(summary.required_bucket_capacity, bucket_hint),
                    std::numeric_limits<uint32_t>::max()));
            gs::detail::updateAtomicMax(
                stats.overflow_required_bytes,
                static_cast<uint64_t>(required_instances)
                        * (2u * sizeof(KeyT) + 2u * sizeof(uint))
                    + static_cast<uint64_t>(required_buckets)
                        * config::block_size_blend * (sizeof(uint) + sizeof(float4)));
            instance_capacity = reserve_instance_capacity(required_instances);
            bucket_capacity = reserve_bucket_capacity(required_buckets);
            continue;
        }

        throwFasterValidationError(summary.error_flags);
        if (summary.n_instances > static_cast<uint32_t>(std::numeric_limits<int>::max())
            || summary.n_buckets > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("FasterGS packed counts exceed int range");
        }

		char* bucket_blob = resize_bucket_buffers(required<BucketBuffers>(bucket_capacity));
		auto bucket_buffers = BucketBuffers::from_blob(bucket_blob, bucket_capacity);
		float4* bucket_geometry = nullptr;
		if (render_geometry) {
			char* geometry_bucket_blob = resize_geometry_bucket_buffers(
				required<GeometryBucketBuffers>(bucket_capacity));
			bucket_geometry = GeometryBucketBuffers::from_blob(
				geometry_bucket_blob, bucket_capacity).depth_normal;
		}
#define FASTER_BLEND_ARGS                                                         \
			tile_buffers.instance_ranges,                                             \
			tile_buffers.buckets_offset,                                              \
			instance_buffers.primitive_indices.Current(),                             \
			primitive_buffers.mean2d,                                                 \
			primitive_buffers.conic_opacity,                                          \
			primitive_buffers.color,                                                  \
			primitive_planes,                                                         \
			bg_color,                                                                 \
			image,                                                                    \
			tile_buffers.final_transmittances,                                        \
			tile_buffers.max_n_processed,                                             \
			tile_buffers.n_processed,                                                 \
			bucket_buffers.tile_index,                                                \
			bucket_buffers.color_transmittance,                                       \
			bucket_geometry,                                                          \
			depth_image,                                                              \
			alpha_image,                                                              \
			normal_image,                                                             \
			width,                                                                    \
			height,                                                                   \
			grid.x,                                                                  \
			focal_x,                                                                 \
			focal_y,                                                                 \
			center_x,                                                                \
			center_y
		if (render_geometry) {
			kernels::forward::blend_cu<true><<<grid, block, 0, stream>>>(
				FASTER_BLEND_ARGS);
		}
		else {
			kernels::forward::blend_cu<false><<<grid, block, 0, stream>>>(
				FASTER_BLEND_ARGS);
		}
#undef FASTER_BLEND_ARGS
        CHECK_CUDA(config::debug, "blend")
        return {static_cast<int>(summary.n_instances),
                static_cast<int>(summary.n_buckets),
                static_cast<int>(instance_capacity),
                static_cast<int>(bucket_capacity),
                selector};
    }
    throw std::runtime_error("unreachable FasterGS capacity retry state");
}

} // namespace

std::tuple<int, int, int, int, int> faster_gs::rasterization::forward(
    std::function<char*(size_t)> resize_primitive_buffers,
    std::function<char*(size_t)> resize_tile_buffers,
    std::function<char*(size_t)> resize_instance_buffers,
    std::function<char*(size_t)> resize_bucket_buffers,
    std::function<char*(size_t)> resize_geometry_primitive_buffers,
    std::function<char*(size_t)> resize_geometry_bucket_buffers,
    std::function<uint32_t(uint32_t)> reserve_instance_capacity,
    std::function<uint32_t(uint32_t)> reserve_bucket_capacity,
    uint32_t initial_instance_capacity,
    uint32_t initial_bucket_capacity,
    RasterCountSummary* summary_device,
    std::function<const RasterCountSummary&()> read_summary,
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
    const bool render_geometry)
{
    GOFUSION_NVTX_RANGE("FasterGS/kernels/forward_packed");
    const auto stream_object = c10::cuda::getCurrentCUDAStream();
    const cudaStream_t stream = stream_object.stream();
    const dim3 grid(div_round_up(width, config::tile_width),
                    div_round_up(height, config::tile_height),
                    1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = static_cast<int>(grid.x * grid.y);
    const int end_bit = tile_key_end_bit(static_cast<uint>(n_tiles));

    char* tile_blob = resize_tile_buffers(required<TileBuffers>(n_tiles));
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_blob, n_tiles);
    char* primitive_blob =
        resize_primitive_buffers(required<PrimitiveBuffers>(n_primitives));
    PrimitiveBuffers primitive_buffers =
        PrimitiveBuffers::from_blob(primitive_blob, n_primitives);
    float4* primitive_planes = nullptr;
    if (render_geometry) {
        char* geometry_primitive_blob = resize_geometry_primitive_buffers(
            required<GeometryPrimitiveBuffers>(n_primitives));
        primitive_planes = GeometryPrimitiveBuffers::from_blob(
            geometry_primitive_blob, n_primitives).planes;
    }

    cudaMemsetAsync(tile_buffers.validation_error, 0, sizeof(uint), stream);
    cudaMemsetAsync(primitive_buffers.n_visible_primitives, 0, sizeof(uint), stream);
    cudaMemsetAsync(primitive_buffers.n_instances, 0, sizeof(uint), stream);
    if (n_primitives > 0) {
        cudaMemsetAsync(primitive_buffers.depth_keys.Current(),
                        0xff,
                        static_cast<size_t>(n_primitives) * sizeof(uint),
                        stream);
        cudaMemsetAsync(primitive_buffers.primitive_indices.Current(),
                        0xff,
                        static_cast<size_t>(n_primitives) * sizeof(uint),
                        stream);
        kernels::forward::preprocess_cu<<<
            div_round_up(n_primitives, config::block_size_preprocess),
            config::block_size_preprocess,
            0,
            stream>>>(means,
                      scales,
                      rotations,
                      opacities,
                      sh_coefficients_0,
                      sh_coefficients_rest,
                      w2c,
                      cam_position,
                      primitive_buffers.depth_keys.Current(),
                      primitive_buffers.primitive_indices.Current(),
                      primitive_buffers.n_touched_tiles,
                      primitive_buffers.screen_bounds,
                      primitive_buffers.mean2d,
                      primitive_buffers.conic_opacity,
                      primitive_buffers.color,
                      primitive_planes,
                      primitive_buffers.n_visible_primitives,
                      primitive_buffers.n_instances,
                      n_primitives,
                      grid.x,
                      grid.y,
                      active_sh_bases,
                      total_sh_bases,
                      static_cast<float>(width),
                      static_cast<float>(height),
                      focal_x,
                      focal_y,
                      center_x,
                      center_y,
                      near_plane,
                      far_plane,
                      proper_antialiasing);
    }
    CHECK_CUDA(config::debug, "preprocess")

    if (n_primitives > 0) {
        cub::DeviceRadixSort::SortPairs(primitive_buffers.cub_workspace,
                                        primitive_buffers.cub_workspace_size,
                                        primitive_buffers.depth_keys,
                                        primitive_buffers.primitive_indices,
                                        n_primitives,
                                        0,
                                        sizeof(uint) * 8,
                                        stream);
        kernels::forward::apply_depth_ordering_capacity_cu<<<
            div_round_up(n_primitives, config::block_size_apply_depth_ordering),
            config::block_size_apply_depth_ordering,
            0,
            stream>>>(primitive_buffers.primitive_indices.Current(),
                      primitive_buffers.n_touched_tiles,
                      primitive_buffers.offset,
                      primitive_buffers.n_visible_primitives,
                      n_primitives);
        cub::DeviceScan::ExclusiveSum(primitive_buffers.cub_workspace,
                                      primitive_buffers.cub_workspace_size,
                                      primitive_buffers.offset,
                                      primitive_buffers.offset,
                                      n_primitives,
                                      stream);
    }
    CHECK_CUDA(config::debug, "depth sort and scan capacity")

    // Bootstrap each persistent scratch slot with the exact device count.
    // This costs one additional synchronization only on the slot's first use
    // and avoids permanently sorting an overly conservative P-based guess.
    if (initial_instance_capacity == 0) {
        packFasterBootstrapSummary<<<1, 1, 0, stream>>>(
            summary_device,
            primitive_buffers.n_visible_primitives,
            primitive_buffers.n_instances,
            n_primitives,
            n_tiles);
        const RasterCountSummary bootstrap = read_summary();
        if (bootstrap.error_flags != 0)
            throw std::runtime_error("FasterGS bootstrap produced invalid counts");
        initial_instance_capacity = reserve_instance_capacity(
            std::max<uint32_t>(bootstrap.n_instances, 1u));
        initial_bucket_capacity = reserve_bucket_capacity(initialBucketCapacity(
            initial_instance_capacity, n_tiles));
    }

#define CAPACITY_RASTER_ARGS                                                        \
    resize_instance_buffers, resize_bucket_buffers, resize_geometry_bucket_buffers, \
        reserve_instance_capacity,                                                 \
        reserve_bucket_capacity, initial_instance_capacity, initial_bucket_capacity, \
        summary_device, read_summary, primitive_buffers, tile_buffers,              \
        primitive_planes, grid, block, bg_color, image, depth_image, alpha_image,   \
        normal_image, n_primitives, n_tiles, end_bit, width, height, focal_x,       \
        focal_y, center_x, center_y, render_geometry, stream
    if (end_bit <= 16)
        return rasterizeCapacityAware<ushort>(CAPACITY_RASTER_ARGS);
    return rasterizeCapacityAware<uint>(CAPACITY_RASTER_ARGS);
#undef CAPACITY_RASTER_ARGS
}
