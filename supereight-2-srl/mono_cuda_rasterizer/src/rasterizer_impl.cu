/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#include "mono_rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <c10/cuda/CUDAStream.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "mono_auxiliary.h"
#include "mono_forward.h"
#include "mono_backward.h"
#include "gs/raster_count_summary.cuh"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t mono_getHigherMsb_impl(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Collect visible Gaussians and corresponding depth keys.
__global__ void mono_collectVisibleGaussians_impl(
	int P,
	const float* depths,
	const uint32_t* tiles_touched,
	uint32_t* visible_depth_keys_unsorted,
	uint32_t* visible_point_indices_unsorted,
	uint32_t* n_visible)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= P) {
		return;
	}
	if (tiles_touched[idx] == 0) {
		return;
	}

	const uint32_t offset = atomicAdd(n_visible, 1U);
	visible_depth_keys_unsorted[offset] = __float_as_uint(depths[idx]);
	visible_point_indices_unsorted[offset] = static_cast<uint32_t>(idx);
}

// Compute the number of touched tiles for depth-sorted visible Gaussians.
__global__ void mono_applyDepthOrdering_impl(
	int n_visible,
	const uint32_t* visible_point_indices_sorted,
	const uint32_t* tiles_touched,
	uint32_t* visible_tiles_touched_sorted)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= n_visible) {
		return;
	}
	const uint32_t point_idx = visible_point_indices_sorted[idx];
	visible_tiles_touched_sorted[idx] = tiles_touched[point_idx];
}

__global__ void mono_applyDepthOrderingCapacity_impl(
	int primitive_capacity,
	const uint32_t* n_visible,
	const uint32_t* visible_point_indices_sorted,
	const uint32_t* tiles_touched,
	uint32_t* visible_tiles_touched_sorted)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= primitive_capacity) return;
	const uint32_t visible = min(*n_visible, static_cast<uint32_t>(primitive_capacity));
	visible_tiles_touched_sorted[idx] = idx < visible
		? tiles_touched[visible_point_indices_sorted[idx]]
		: 0U;
}

__global__ void monoComputeRenderedCount_impl(
	int primitive_capacity,
	const uint32_t* visible_offsets,
	const uint32_t* visible_tiles_touched,
	uint32_t* n_instances)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	*n_instances = primitive_capacity == 0
		? 0U
		: visible_offsets[primitive_capacity - 1]
			+ visible_tiles_touched[primitive_capacity - 1];
}

// Generate one key/value pair for all Gaussian / tile overlaps.
// Run once per depth-sorted visible Gaussian (1:N mapping).
template <typename KeyT>
__global__ void mono_duplicateWithKeys_impl(
	int n_visible,
	const uint32_t* visible_point_indices_sorted,
	const uint32_t* visible_offsets,
	const uint32_t* expected_tiles_touched,
	const float2* points_xy,
	KeyT* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const float4* con_o,
	dim3 grid,
	uint32_t* validation_error,
	uint32_t* validation_details)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= n_visible) {
		return;
	}

	const uint32_t point_idx = visible_point_indices_sorted[idx];
	const uint32_t off = visible_offsets[idx];
	const uint32_t expected = expected_tiles_touched[point_idx];
	const uint32_t actual = mono_duplicateToTilesTouchedByTileKey_aux<KeyT>(
		points_xy[point_idx],
		con_o[point_idx],
		grid,
		point_idx,
		off,
		expected,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
	if (actual != expected) {
		atomicOr(validation_error, 4U);
		if (atomicCAS(validation_details, UINT32_MAX, 4U) == UINT32_MAX) {
			validation_details[1] = point_idx;
			validation_details[2] = expected;
			validation_details[3] = actual;
		}
	}
}

template <typename KeyT>
__global__ void mono_fillInstancePadding_impl(
	uint32_t count,
	KeyT padding_key,
	KeyT* keys,
	uint32_t* values)
{
	const uint32_t idx = cg::this_grid().thread_rank();
	if (idx >= count) return;
	keys[idx] = padding_key;
	values[idx] = UINT32_MAX;
}

template <typename KeyT>
__global__ void mono_duplicateWithKeysCapacity_impl(
	int primitive_capacity,
	const uint32_t* n_visible,
	const uint32_t* visible_point_indices_sorted,
	const uint32_t* visible_offsets,
	const uint32_t* expected_tiles_touched,
	const float2* points_xy,
	KeyT* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const float4* con_o,
	dim3 grid,
	uint32_t instance_capacity,
	uint32_t* validation_error,
	uint32_t* validation_details)
{
	const auto idx = cg::this_grid().thread_rank();
	const uint32_t visible = min(*n_visible, static_cast<uint32_t>(primitive_capacity));
	if (idx >= visible) return;

	const uint32_t point_idx = visible_point_indices_sorted[idx];
	const uint32_t off = visible_offsets[idx];
	const uint32_t expected = expected_tiles_touched[point_idx];
	const uint32_t writable = off < instance_capacity
		? min(expected, instance_capacity - off)
		: 0U;
	if (expected > writable) {
		atomicOr(validation_error, gs::detail::kRasterOverflowInstances);
	}
	const uint32_t actual = mono_duplicateToTilesTouchedByTileKey_aux<KeyT>(
		points_xy[point_idx],
		con_o[point_idx],
		grid,
		point_idx,
		off,
		writable,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
	if (actual != expected) {
		atomicOr(validation_error, 4U);
		if (atomicCAS(validation_details, UINT32_MAX, 4U) == UINT32_MAX) {
			validation_details[1] = point_idx;
			validation_details[2] = expected;
			validation_details[3] = actual;
		}
	}
}

template <typename KeyT>
__global__ void mono_validateSortedInstancesCapacity_impl(
	uint32_t instance_capacity,
	const uint32_t* n_instances,
	int n_tiles,
	int n_points,
	const KeyT* point_list_keys,
	const uint32_t* point_list,
	uint32_t* validation_error,
	uint32_t* validation_details)
{
	const auto idx = cg::this_grid().thread_rank();
	const uint32_t effective = min(*n_instances, instance_capacity);
	if (idx >= effective) return;
	const uint32_t tile = static_cast<uint32_t>(point_list_keys[idx]);
	if (tile >= static_cast<uint32_t>(n_tiles)) {
		atomicOr(validation_error, 1U);
		if (atomicCAS(validation_details, UINT32_MAX, 1U) == UINT32_MAX) {
			validation_details[1] = idx;
			validation_details[2] = tile;
			validation_details[3] = n_tiles;
		}
		return;
	}
	if (idx > 0 && point_list_keys[idx - 1] > point_list_keys[idx]) {
		atomicOr(validation_error, 8U);
	}
	const uint32_t point_idx = point_list[idx];
	if (point_idx >= static_cast<uint32_t>(n_points)) {
		atomicOr(validation_error, 16U);
		if (atomicCAS(validation_details, UINT32_MAX, 16U) == UINT32_MAX) {
			validation_details[1] = idx;
			validation_details[2] = point_idx;
			validation_details[3] = n_points;
		}
	}
}

template <typename KeyT>
__global__ void mono_buildTileRangesCapacity_impl(
	uint32_t instance_capacity,
	const uint32_t* n_instances,
	int n_tiles,
	const KeyT* point_list_keys,
	uint2* ranges)
{
	const uint32_t tile_idx = cg::this_grid().thread_rank();
	if (tile_idx >= static_cast<uint32_t>(n_tiles)) return;
	const uint32_t effective = min(*n_instances, instance_capacity);
	uint32_t lower_first = 0;
	uint32_t lower_count = effective;
	while (lower_count > 0) {
		const uint32_t step = lower_count / 2;
		const uint32_t probe = lower_first + step;
		if (static_cast<uint32_t>(point_list_keys[probe]) < tile_idx) {
			lower_first = probe + 1;
			lower_count -= step + 1;
		} else lower_count = step;
	}
	uint32_t upper_first = lower_first;
	uint32_t upper_count = effective - upper_first;
	while (upper_count > 0) {
		const uint32_t step = upper_count / 2;
		const uint32_t probe = upper_first + step;
		if (static_cast<uint32_t>(point_list_keys[probe]) <= tile_idx) {
			upper_first = probe + 1;
			upper_count -= step + 1;
		} else upper_count = step;
	}
	ranges[tile_idx] = make_uint2(lower_first, upper_first);
}

__global__ void monoPackRasterSummary_impl(
	gs::detail::RasterCountSummary* summary,
	const uint32_t* n_visible,
	const uint32_t* n_instances,
	const uint32_t* bucket_offsets,
	const uint32_t* validation_error,
	const uint32_t* validation_details,
	uint32_t n_primitives,
	uint32_t n_tiles,
	uint32_t instance_capacity,
	uint32_t bucket_capacity)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	gs::detail::RasterCountSummary result{};
	result.n_visible = *n_visible;
	result.n_instances = *n_instances;
	result.n_buckets = n_tiles == 0 ? 0U : bucket_offsets[n_tiles - 1];
	result.error_flags = *validation_error;
	result.required_instance_capacity = result.n_instances;
	result.required_bucket_capacity = result.n_buckets;
	result.detail0 = validation_details == nullptr ? 0U : validation_details[0];
	result.detail1 = validation_details == nullptr ? 0U : validation_details[1];
	const uint64_t maximum =
		static_cast<uint64_t>(result.n_visible) * static_cast<uint64_t>(n_tiles);
	if (result.n_visible > n_primitives
		|| static_cast<uint64_t>(result.n_instances) > maximum
		|| result.n_buckets > result.n_instances)
		result.error_flags |= gs::detail::kRasterInvalidCounts;
	if (result.n_instances > instance_capacity)
		result.error_flags |= gs::detail::kRasterOverflowInstances;
	if (result.n_buckets > bucket_capacity)
		result.error_flags |= gs::detail::kRasterOverflowBuckets;
	*summary = result;
}

__global__ void monoPackBootstrapSummary_impl(
	gs::detail::RasterCountSummary* summary,
	const uint32_t* n_visible,
	const uint32_t* n_instances,
	uint32_t n_primitives,
	uint32_t n_tiles)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	gs::detail::RasterCountSummary result{};
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

// Validate the complete sorted instance stream before any key is used as an
// array index. This also verifies that the radix sort produced a monotonic
// sequence and that every value still refers to a valid Gaussian.
template <typename KeyT>
__global__ void mono_validateSortedInstances_impl(
	int L,
	int n_tiles,
	int n_points,
	const KeyT* point_list_keys,
	const uint32_t* point_list,
	uint32_t* validation_error,
	uint32_t* validation_details)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= L) {
		return;
	}

	const uint32_t currtile = static_cast<uint32_t>(point_list_keys[idx]);
	if (currtile >= static_cast<uint32_t>(n_tiles)) {
		atomicOr(validation_error, 1U);
		if (atomicCAS(validation_details, UINT32_MAX, 1U) == UINT32_MAX) {
			validation_details[1] = static_cast<uint32_t>(idx);
			validation_details[2] = currtile;
			validation_details[3] = static_cast<uint32_t>(n_tiles);
		}
		return;
	}

	if (idx > 0) {
		const uint32_t prevtile = static_cast<uint32_t>(point_list_keys[idx - 1]);
		if (prevtile > currtile) {
			atomicOr(validation_error, 8U);
			if (atomicCAS(validation_details, UINT32_MAX, 8U) == UINT32_MAX) {
				validation_details[1] = static_cast<uint32_t>(idx);
				validation_details[2] = prevtile;
				validation_details[3] = currtile;
			}
		}
	}

	const uint32_t point_idx = point_list[idx];
	if (point_idx >= static_cast<uint32_t>(n_points)) {
		atomicOr(validation_error, 16U);
		if (atomicCAS(validation_details, UINT32_MAX, 16U) == UINT32_MAX) {
			validation_details[1] = static_cast<uint32_t>(idx);
			validation_details[2] = point_idx;
			validation_details[3] = static_cast<uint32_t>(n_points);
		}
	}
}

// Construct each tile range independently with lower/upper bounds. Compared
// with adjacent-instance writes, this is deterministic and has no cross-thread
// writes into the same uint2.
template <typename KeyT>
__global__ void mono_buildTileRanges_impl(
	int L,
	int n_tiles,
	const KeyT* point_list_keys,
	uint2* ranges)
{
	const auto tile_idx = cg::this_grid().thread_rank();
	if (tile_idx >= n_tiles) {
		return;
	}

	uint32_t lower_first = 0;
	uint32_t lower_count = static_cast<uint32_t>(L);
	while (lower_count > 0) {
		const uint32_t step = lower_count / 2;
		const uint32_t probe = lower_first + step;
		if (static_cast<uint32_t>(point_list_keys[probe])
			< static_cast<uint32_t>(tile_idx)) {
			lower_first = probe + 1;
			lower_count -= step + 1;
		} else {
			lower_count = step;
		}
	}

	uint32_t upper_first = lower_first;
	uint32_t upper_count = static_cast<uint32_t>(L) - upper_first;
	while (upper_count > 0) {
		const uint32_t step = upper_count / 2;
		const uint32_t probe = upper_first + step;
		if (static_cast<uint32_t>(point_list_keys[probe])
			<= static_cast<uint32_t>(tile_idx)) {
			upper_first = probe + 1;
			upper_count -= step + 1;
		} else {
			upper_count = step;
		}
	}

	ranges[tile_idx] = make_uint2(lower_first, upper_first);
}

__global__ void mono_extractBucketCounts_impl(
	int n_tiles,
	const uint32_t* n_instances,
	uint32_t instance_capacity,
	const uint2* tile_ranges,
	uint32_t* tile_n_buckets,
	uint32_t* validation_error,
	uint32_t* validation_details)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= n_tiles) {
		return;
	}
	const uint2 range = tile_ranges[idx];
	const uint32_t effective = min(*n_instances, instance_capacity);
	if (range.x > range.y || range.y > effective) {
		tile_n_buckets[idx] = 0;
		atomicOr(validation_error, 2U);
		if (atomicCAS(validation_details, UINT32_MAX, 2U) == UINT32_MAX) {
			validation_details[1] = static_cast<uint32_t>(idx);
			validation_details[2] = range.x;
			validation_details[3] = range.y;
		}
		return;
	}
	const uint32_t range_instances = range.y - range.x;
	tile_n_buckets[idx] = (range_instances + 31U) / 32U;
}

MonoCudaRasterizer::GeometryState MonoCudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	const size_t visible_query_count = std::max<size_t>(1, P);
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	obtain(chunk, geom.n_visible, 1, 128);
	obtain(chunk, geom.n_instances, 1, 128);
	obtain(chunk, geom.visible_point_indices, P, 128);
	obtain(chunk, geom.visible_point_indices_unsorted, P, 128);
	obtain(chunk, geom.visible_depth_keys, P, 128);
	obtain(chunk, geom.visible_depth_keys_unsorted, P, 128);
	obtain(chunk, geom.visible_tiles_touched, P, 128);
	obtain(chunk, geom.visible_offsets, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, geom.depth_sorting_size,
		geom.visible_depth_keys_unsorted, geom.visible_depth_keys,
		geom.visible_point_indices_unsorted, geom.visible_point_indices,
		visible_query_count);
	obtain(chunk, geom.depth_sorting_space, geom.depth_sorting_size, 128);
	cub::DeviceScan::ExclusiveSum(
		nullptr, geom.visible_scan_size,
		geom.visible_tiles_touched, geom.visible_offsets,
		visible_query_count);
	obtain(chunk, geom.visible_scan_space, geom.visible_scan_size, 128);
	return geom;
}

MonoCudaRasterizer::ImageState MonoCudaRasterizer::ImageState::fromChunks(
	char*& fixed_chunk,
	char*& bucket_chunk,
	size_t N,
	size_t T,
	size_t B)
{
	ImageState img;
	obtain(fixed_chunk, img.accum_alpha, N, 128);
	obtain(fixed_chunk, img.n_contrib, N, 128);
	obtain(fixed_chunk, img.tile_max_n_processed, T, 128);
	obtain(bucket_chunk, img.bucket_tile_index, B, 128);
	obtain(bucket_chunk, img.bucket_color_transmittance, B * BLOCK_SIZE, 128);
	obtain(bucket_chunk, img.bucket_depth_accum, B * BLOCK_SIZE, 128);
	return img;
}

MonoCudaRasterizer::BinningState MonoCudaRasterizer::BinningState::fromChunks(
	char*& tile_chunk, char*& instance_chunk, size_t R, size_t T)
{
	BinningState binning;
	const size_t instance_query_count = std::max<size_t>(1, R);
	const size_t tile_query_count = std::max<size_t>(1, T);

	obtain(tile_chunk, binning.tile_ranges, T, 128);
	obtain(tile_chunk, binning.tile_n_buckets, T, 128);
	obtain(tile_chunk, binning.tile_bucket_offsets, T, 128);
	obtain(tile_chunk, binning.validation_error, 1, 128);
	obtain(tile_chunk, binning.validation_details, 4, 128);
	cub::DeviceScan::InclusiveSum(
		nullptr, binning.bucket_scan_size,
		binning.tile_n_buckets, binning.tile_bucket_offsets,
		tile_query_count);
	obtain(tile_chunk, binning.bucket_scan_space, binning.bucket_scan_size, 128);

	obtain(instance_chunk, binning.point_list, R, 128);
	obtain(instance_chunk, binning.point_list_unsorted, R, 128);
	const uint32_t tile_key_bits = mono_getHigherMsb_impl(
		static_cast<uint32_t>(tile_query_count - 1));
	binning.tile_sort_end_bit = static_cast<int>(tile_key_bits + 1U);
	binning.padding_key = 1U << tile_key_bits;
	binning.use_u16_tile_keys = binning.tile_sort_end_bit <= 16;
	if (binning.use_u16_tile_keys) {
		binning.point_list_keys_u32 = nullptr;
		binning.point_list_keys_u32_unsorted = nullptr;
		obtain(instance_chunk, binning.point_list_keys_u16, R, 128);
		obtain(instance_chunk, binning.point_list_keys_u16_unsorted, R, 128);
		cub::DeviceRadixSort::SortPairs(
			nullptr, binning.tile_sorting_size,
			binning.point_list_keys_u16_unsorted, binning.point_list_keys_u16,
			binning.point_list_unsorted, binning.point_list,
			instance_query_count, 0, binning.tile_sort_end_bit);
	} else {
		binning.point_list_keys_u16 = nullptr;
		binning.point_list_keys_u16_unsorted = nullptr;
		obtain(instance_chunk, binning.point_list_keys_u32, R, 128);
		obtain(instance_chunk, binning.point_list_keys_u32_unsorted, R, 128);
		cub::DeviceRadixSort::SortPairs(
			nullptr, binning.tile_sorting_size,
			binning.point_list_keys_u32_unsorted, binning.point_list_keys_u32,
			binning.point_list_unsorted, binning.point_list,
			instance_query_count, 0, binning.tile_sort_end_bit);
	}
	obtain(instance_chunk, binning.tile_sorting_space, binning.tile_sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
namespace {

uint32_t monoInitialInstanceCapacity(const int point_count)
{
	(void) point_count;
	return 4096U;
}

uint32_t monoInitialBucketCapacity(const uint32_t instance_capacity, const int n_tiles)
{
	return static_cast<uint32_t>(std::min<uint64_t>(
		std::max<uint64_t>(1024U,
			static_cast<uint64_t>(instance_capacity) / 16U
				+ static_cast<uint64_t>(std::max(0, n_tiles))),
		static_cast<uint64_t>(std::numeric_limits<int>::max())));
}

class MonoPackedFallback final : public std::runtime_error
{
public:
	using std::runtime_error::runtime_error;
};

std::string monoPackedValidationReason(
	const gs::detail::RasterCountSummary& summary)
{
	if ((summary.error_flags & gs::detail::kRasterInvalidCounts) != 0)
		return "Mono rasterizer produced invalid packed counts";
	const uint32_t kind = summary.detail0;
	if (kind == 1U) return "Mono rasterizer produced an invalid tile key";
	if (kind == 2U) return "Mono rasterizer produced an invalid tile range";
	if (kind == 4U) return "Mono rasterizer count/write mismatch";
	if (kind == 8U) return "Mono rasterizer tile sort is not monotonic";
	if (kind == 16U) return "Mono rasterizer produced an invalid Gaussian index";
	return "Mono rasterizer packed validation failed, flags="
		+ std::to_string(summary.error_flags);
}

int monoForwardDeviceSummary(
	std::function<char*(size_t)> geometryBuffer,
	std::function<char*(size_t)> tileBuffer,
	std::function<char*(size_t)> instanceBuffer,
	std::function<char*(size_t)> imageFixedBuffer,
	std::function<char*(size_t)> bucketBuffer,
	std::function<char*(size_t)> geometryPlaneBuffer,
	std::function<char*(size_t)> geometryBucketBuffer,
	std::function<uint32_t(uint32_t)> reserveInstanceCapacity,
	std::function<uint32_t(uint32_t)> reserveBucketCapacity,
	uint32_t initialInstanceCapacity,
	uint32_t initialBucketCapacity,
	gs::detail::RasterCountSummary* summaryDevice,
	std::function<const gs::detail::RasterCountSummary&()> readSummary,
	const int P,
	int D,
	int M,
	const float* background,
	const int width,
	int height,
	const float* means3D,
	const float* shs_dc,
	const float* shs_rest,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx,
	float tan_fovy,
	const float focal_x,
	float focal_y,
	const float center_x,
	float center_y,
	const bool prefiltered,
	const bool render_geometry,
	float* out_color,
	float* out_depth,
	float* out_opacity,
	float* out_normal,
	int* radii,
	int* n_touched,
	int* n_buckets_out,
	int* instance_capacity_out,
	int* bucket_capacity_out,
	bool debug,
	const bool capacity_sort)
{
	using namespace MonoCudaRasterizer;
	const auto stream_object = c10::cuda::getCurrentCUDAStream();
	const cudaStream_t stream = stream_object.stream();
	auto check_cub = [&](const cudaError_t status, const char* operation) {
		if (status == cudaSuccess) return;
		const std::string message = std::string("Mono rasterizer ") + operation
			+ " failed: " + cudaGetErrorString(status);
		if (capacity_sort) throw MonoPackedFallback(message);
		throw std::runtime_error(message);
	};
	char* geometry_ptr = geometryBuffer(required<GeometryState>(P));
	GeometryState geometry = GeometryState::fromChunk(geometry_ptr, P);
	float4* geometry_planes = nullptr;
	if (render_geometry) {
		char* plane_ptr = geometryPlaneBuffer(required_geometry_planes(P));
		geometry_planes = geometry_planes_from_chunk(plane_ptr, P);
	}
	if (radii == nullptr) radii = geometry.internal_radii;
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int n_tiles = static_cast<int>(tile_grid.x * tile_grid.y);
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");

	MONO_FORWARD::preprocess(P,
		D,
		M,
		means3D,
		reinterpret_cast<const glm::vec3*>(scales),
		scale_modifier,
		reinterpret_cast<const glm::vec4*>(rotations),
		opacities,
		shs_dc,
		shs_rest,
		geometry.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		reinterpret_cast<const glm::vec3*>(cam_pos),
		width,
		height,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		radii,
		geometry.means2D,
		geometry.depths,
		geometry.cov3D,
		geometry.rgb,
		geometry.conic_opacity,
		geometry_planes,
		tile_grid,
		geometry.tiles_touched,
		prefiltered,
		stream);
	MONO_CHECK_CUDA(, debug)

	cudaMemsetAsync(geometry.n_visible, 0, sizeof(uint32_t), stream);
	cudaMemsetAsync(geometry.n_instances, 0, sizeof(uint32_t), stream);
	if (P > 0) {
		cudaMemsetAsync(geometry.visible_depth_keys_unsorted,
			0xff,
			static_cast<size_t>(P) * sizeof(uint32_t),
			stream);
		cudaMemsetAsync(geometry.visible_point_indices_unsorted,
			0xff,
			static_cast<size_t>(P) * sizeof(uint32_t),
			stream);
		mono_collectVisibleGaussians_impl<<<(P + 255) / 256, 256, 0, stream>>>(
			P,
			geometry.depths,
			geometry.tiles_touched,
			geometry.visible_depth_keys_unsorted,
			geometry.visible_point_indices_unsorted,
			geometry.n_visible);
		check_cub(cub::DeviceRadixSort::SortPairs(geometry.depth_sorting_space,
			geometry.depth_sorting_size,
			geometry.visible_depth_keys_unsorted,
			geometry.visible_depth_keys,
			geometry.visible_point_indices_unsorted,
			geometry.visible_point_indices,
			P,
			0,
			32,
			stream), "depth radix sort");
		mono_applyDepthOrderingCapacity_impl<<<(P + 255) / 256, 256, 0, stream>>>(
			P,
			geometry.n_visible,
			geometry.visible_point_indices,
			geometry.tiles_touched,
			geometry.visible_tiles_touched);
		check_cub(cub::DeviceScan::ExclusiveSum(geometry.visible_scan_space,
			geometry.visible_scan_size,
			geometry.visible_tiles_touched,
			geometry.visible_offsets,
			P,
			stream), "visible instance scan");
	}
	monoComputeRenderedCount_impl<<<1, 1, 0, stream>>>(
		P, geometry.visible_offsets, geometry.visible_tiles_touched, geometry.n_instances);
	MONO_CHECK_CUDA(, debug)

	auto run_device_summary = [&]() -> int {
		uint32_t exact_instances = 0;
		if (!capacity_sort) {
			monoPackBootstrapSummary_impl<<<1, 1, 0, stream>>>(
				summaryDevice, geometry.n_visible, geometry.n_instances, P, n_tiles);
			MONO_CHECK_CUDA(, debug)
			const auto bootstrap = readSummary();
			if (bootstrap.error_flags != 0
				|| bootstrap.n_instances
					> static_cast<uint32_t>(std::numeric_limits<int>::max())) {
				throw std::runtime_error(
					"Mono exact raster bootstrap produced invalid counts");
			}
			exact_instances = bootstrap.n_instances;
		}

		uint32_t instance_capacity = capacity_sort
			? reserveInstanceCapacity(std::max(
				initialInstanceCapacity, monoInitialInstanceCapacity(P)))
			: reserveInstanceCapacity(std::max(exact_instances, 1U));
		uint32_t bucket_capacity = initialBucketCapacity;
		if (bucket_capacity == 0) {
			bucket_capacity = reserveBucketCapacity(
				monoInitialBucketCapacity(instance_capacity, n_tiles));
		}

		char* tile_base = tileBuffer(required_tile(static_cast<size_t>(n_tiles)));
		const size_t pixel_count =
			static_cast<size_t>(width) * static_cast<size_t>(height);
		char* image_fixed_base = imageFixedBuffer(
			required_image_fixed(pixel_count, static_cast<size_t>(n_tiles)));

		for (int attempt = 0; attempt < 2; ++attempt) {
			const uint32_t sort_count = capacity_sort ? instance_capacity : exact_instances;
			char* tile_ptr = tile_base;
			char* instance_ptr = instanceBuffer(required_instance(
				instance_capacity, static_cast<size_t>(n_tiles)));
			BinningState binning = BinningState::fromChunks(
				tile_ptr, instance_ptr, instance_capacity,
				static_cast<size_t>(n_tiles));
			cudaMemsetAsync(binning.validation_error, 0, sizeof(uint32_t), stream);
			cudaMemsetAsync(
				binning.validation_details, 0xff, 4 * sizeof(uint32_t), stream);

			if (sort_count > 0) {
				const dim3 fill_grid((sort_count + 255U) / 256U);
				if (binning.use_u16_tile_keys) {
					mono_fillInstancePadding_impl<uint16_t><<<fill_grid, 256, 0, stream>>>(
						sort_count,
						static_cast<uint16_t>(binning.padding_key),
						binning.point_list_keys_u16_unsorted,
						binning.point_list_unsorted);
				} else {
					mono_fillInstancePadding_impl<uint32_t><<<fill_grid, 256, 0, stream>>>(
						sort_count,
						binning.padding_key,
						binning.point_list_keys_u32_unsorted,
						binning.point_list_unsorted);
				}
			}

			if (P > 0) {
				if (binning.use_u16_tile_keys) {
					mono_duplicateWithKeysCapacity_impl<uint16_t><<<
						(P + 255) / 256, 256, 0, stream>>>(
						P, geometry.n_visible, geometry.visible_point_indices,
						geometry.visible_offsets, geometry.tiles_touched,
						geometry.means2D, binning.point_list_keys_u16_unsorted,
						binning.point_list_unsorted, geometry.conic_opacity,
						tile_grid, instance_capacity, binning.validation_error,
						binning.validation_details);
				} else {
					mono_duplicateWithKeysCapacity_impl<uint32_t><<<
						(P + 255) / 256, 256, 0, stream>>>(
						P, geometry.n_visible, geometry.visible_point_indices,
						geometry.visible_offsets, geometry.tiles_touched,
						geometry.means2D, binning.point_list_keys_u32_unsorted,
						binning.point_list_unsorted, geometry.conic_opacity,
						tile_grid, instance_capacity, binning.validation_error,
						binning.validation_details);
				}
			}

			if (sort_count > 0) {
				if (binning.use_u16_tile_keys) {
					check_cub(cub::DeviceRadixSort::SortPairs(
						binning.tile_sorting_space, binning.tile_sorting_size,
						binning.point_list_keys_u16_unsorted,
						binning.point_list_keys_u16,
						binning.point_list_unsorted, binning.point_list,
						static_cast<int>(sort_count), 0,
						binning.tile_sort_end_bit, stream), "tile radix sort");
					mono_validateSortedInstancesCapacity_impl<uint16_t><<<
						(sort_count + 255U) / 256U, 256, 0, stream>>>(
						instance_capacity, geometry.n_instances, n_tiles, P,
						binning.point_list_keys_u16, binning.point_list,
						binning.validation_error, binning.validation_details);
				} else {
					check_cub(cub::DeviceRadixSort::SortPairs(
						binning.tile_sorting_space, binning.tile_sorting_size,
						binning.point_list_keys_u32_unsorted,
						binning.point_list_keys_u32,
						binning.point_list_unsorted, binning.point_list,
						static_cast<int>(sort_count), 0,
						binning.tile_sort_end_bit, stream), "tile radix sort");
					mono_validateSortedInstancesCapacity_impl<uint32_t><<<
						(sort_count + 255U) / 256U, 256, 0, stream>>>(
						instance_capacity, geometry.n_instances, n_tiles, P,
						binning.point_list_keys_u32, binning.point_list,
						binning.validation_error, binning.validation_details);
				}
			}

			if (binning.use_u16_tile_keys) {
				mono_buildTileRangesCapacity_impl<uint16_t><<<
					(n_tiles + 255) / 256, 256, 0, stream>>>(
					instance_capacity, geometry.n_instances, n_tiles,
					binning.point_list_keys_u16, binning.tile_ranges);
			} else {
				mono_buildTileRangesCapacity_impl<uint32_t><<<
					(n_tiles + 255) / 256, 256, 0, stream>>>(
					instance_capacity, geometry.n_instances, n_tiles,
					binning.point_list_keys_u32, binning.tile_ranges);
			}
			mono_extractBucketCounts_impl<<<
				(n_tiles + 255) / 256, 256, 0, stream>>>(
				n_tiles, geometry.n_instances, instance_capacity,
				binning.tile_ranges, binning.tile_n_buckets,
				binning.validation_error, binning.validation_details);
			check_cub(cub::DeviceScan::InclusiveSum(
				binning.bucket_scan_space, binning.bucket_scan_size,
				binning.tile_n_buckets, binning.tile_bucket_offsets,
				n_tiles, stream), "bucket scan");

			monoPackRasterSummary_impl<<<1, 1, 0, stream>>>(
				summaryDevice, geometry.n_visible, geometry.n_instances,
				binning.tile_bucket_offsets, binning.validation_error,
				binning.validation_details, P, n_tiles, instance_capacity,
				bucket_capacity);

			char* image_fixed_ptr = image_fixed_base;
			char* bucket_ptr = bucketBuffer(required_bucket(bucket_capacity));
			ImageState image = ImageState::fromChunks(
				image_fixed_ptr, bucket_ptr, pixel_count,
				static_cast<size_t>(n_tiles), bucket_capacity);
			const float* features =
				colors_precomp != nullptr ? colors_precomp : geometry.rgb;
			float4* geometry_bucket_accum = nullptr;
			if (render_geometry) {
				char* geometry_bucket_ptr = geometryBucketBuffer(
					required_geometry_buckets(bucket_capacity));
				geometry_bucket_accum = geometry_buckets_from_chunk(
					geometry_bucket_ptr, bucket_capacity);
			}
			MONO_FORWARD::render(
				tile_grid, block, binning.tile_ranges,
				binning.tile_bucket_offsets, binning.point_list, width, height,
				geometry.means2D, features, geometry.conic_opacity,
				image.accum_alpha, image.n_contrib,
				image.tile_max_n_processed, image.bucket_tile_index,
				image.bucket_color_transmittance, image.bucket_depth_accum,
				geometry_bucket_accum,
				background, out_color, geometry.depths, geometry_planes,
				out_depth, out_opacity, out_normal,
				focal_x, focal_y, center_x, center_y, render_geometry,
				n_touched, &summaryDevice->error_flags, stream);
			MONO_CHECK_CUDA(, debug)

			const auto summary = readSummary();
			auto& stats = gs::detail::rasterDependencyStats();
			gs::detail::updateAtomicMax(stats.instance_high_water, summary.n_instances);
			gs::detail::updateAtomicMax(stats.bucket_high_water, summary.n_buckets);
			gs::detail::maybeReportRasterDependencyStats(
				capacity_sort ? "MonoGS" : "MonoGS-exact",
				summary, instance_capacity, bucket_capacity);

			const uint32_t overflow_mask = gs::detail::kRasterOverflowInstances
				| gs::detail::kRasterOverflowBuckets;
			const uint32_t invariant_errors = summary.error_flags & ~overflow_mask;
			if (invariant_errors != 0) {
				stats.invalid_summaries.fetch_add(1, std::memory_order_relaxed);
				const std::string reason = monoPackedValidationReason(summary);
				if (capacity_sort) throw MonoPackedFallback(reason);
				throw std::runtime_error("Exact fallback failed: " + reason);
			}

			if ((summary.error_flags & gs::detail::kRasterOverflowInstances) != 0) {
				if (!capacity_sort || attempt != 0) {
					if (capacity_sort) {
						throw MonoPackedFallback(
							"Mono instance capacity overflow persisted after one retry");
					}
					throw std::runtime_error("Mono exact instance capacity overflow");
				}
				stats.raster_retries.fetch_add(1, std::memory_order_relaxed);
				gs::detail::updateAtomicMax(
					stats.overflow_required_bytes,
					static_cast<uint64_t>(summary.required_instance_capacity) * 12U);
				instance_capacity = reserveInstanceCapacity(
					std::max(summary.required_instance_capacity,
						instance_capacity + 1U));
				continue;
			}

			if ((summary.error_flags & gs::detail::kRasterOverflowBuckets) != 0) {
				if (summary.n_buckets > summary.n_instances) {
					const std::string reason =
						"Mono bucket count exceeds the exact instance count";
					if (capacity_sort) throw MonoPackedFallback(reason);
					throw std::runtime_error(reason);
				}
				stats.bucket_only_grows.fetch_add(1, std::memory_order_relaxed);
				bucket_capacity = reserveBucketCapacity(summary.required_bucket_capacity);
				image_fixed_ptr = image_fixed_base;
				bucket_ptr = bucketBuffer(required_bucket(bucket_capacity));
				image = ImageState::fromChunks(
					image_fixed_ptr, bucket_ptr, pixel_count,
					static_cast<size_t>(n_tiles), bucket_capacity);
				if (render_geometry) {
					char* geometry_bucket_ptr = geometryBucketBuffer(
						required_geometry_buckets(bucket_capacity));
					geometry_bucket_accum = geometry_buckets_from_chunk(
						geometry_bucket_ptr, bucket_capacity);
				}
				MONO_FORWARD::render(
					tile_grid, block, binning.tile_ranges,
					binning.tile_bucket_offsets, binning.point_list, width, height,
					geometry.means2D, features, geometry.conic_opacity,
					image.accum_alpha, image.n_contrib,
					image.tile_max_n_processed, image.bucket_tile_index,
					image.bucket_color_transmittance, image.bucket_depth_accum,
					geometry_bucket_accum,
					background, out_color, geometry.depths, geometry_planes,
					out_depth, out_opacity, out_normal,
					focal_x, focal_y, center_x, center_y, render_geometry,
					n_touched, nullptr, stream);
				MONO_CHECK_CUDA(, debug)
			}

			if (summary.n_instances
				> static_cast<uint32_t>(std::numeric_limits<int>::max())
				|| summary.n_buckets
					> static_cast<uint32_t>(std::numeric_limits<int>::max())) {
				throw std::runtime_error("Mono raster counts exceed int range");
			}
			if (n_buckets_out) *n_buckets_out = static_cast<int>(summary.n_buckets);
			if (instance_capacity_out) *instance_capacity_out = instance_capacity;
			if (bucket_capacity_out) *bucket_capacity_out = bucket_capacity;
			return static_cast<int>(summary.n_instances);
		}
		throw std::runtime_error("unreachable Mono raster retry state");
	};

	return run_device_summary();
}

} // namespace

int MonoCudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> tileBuffer,
	std::function<char* (size_t)> instanceBuffer,
	std::function<char* (size_t)> imageFixedBuffer,
	std::function<char* (size_t)> bucketBuffer,
	std::function<char* (size_t)> geometryPlaneBuffer,
	std::function<char* (size_t)> geometryBucketBuffer,
	std::function<uint32_t(uint32_t)> reserveInstanceCapacity,
	std::function<uint32_t(uint32_t)> reserveBucketCapacity,
	uint32_t initialInstanceCapacity,
	uint32_t initialBucketCapacity,
	gs::detail::RasterCountSummary* summaryDevice,
	std::function<const gs::detail::RasterCountSummary&()> readSummary,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs_dc,
	const float* shs_rest,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float center_x, float center_y,
	const bool prefiltered,
	const bool render_geometry,
	float* out_color,
	float* out_depth,
	float* out_opacity,
	float* out_normal,
	int* radii,
	int* n_touched,
	int* n_buckets_out,
	int* instance_capacity_out,
	int* bucket_capacity_out,
	bool debug)
{
	auto run = [&](const bool capacity_sort) {
		return monoForwardDeviceSummary(geometryBuffer,
			tileBuffer,
			instanceBuffer,
			imageFixedBuffer,
			bucketBuffer,
			geometryPlaneBuffer,
			geometryBucketBuffer,
			reserveInstanceCapacity,
			reserveBucketCapacity,
			initialInstanceCapacity,
			initialBucketCapacity,
			summaryDevice,
			readSummary,
			P,
			D,
			M,
			background,
			width,
			height,
			means3D,
			shs_dc,
			shs_rest,
			colors_precomp,
			opacities,
			scales,
			scale_modifier,
			rotations,
			cov3D_precomp,
			viewmatrix,
			projmatrix,
			cam_pos,
			tan_fovx,
			tan_fovy,
			focal_x,
			focal_y,
			center_x,
			center_y,
			prefiltered,
			render_geometry,
			out_color,
			out_depth,
			out_opacity,
			out_normal,
			radii,
			n_touched,
			n_buckets_out,
			instance_capacity_out,
			bucket_capacity_out,
			debug,
			capacity_sort);
	};

	if (gs::detail::packedMonoRasterSummaryEnabled()) {
		try {
			return run(true);
		} catch (const MonoPackedFallback& error) {
			auto& stats = gs::detail::rasterDependencyStats();
			stats.mono_exact_fallbacks.fetch_add(1, std::memory_order_relaxed);
			gs::detail::disableMonoPackedRasterRuntime();
			std::cerr << "Mono packed raster validation failed; using exact fallback "
					 "and disabling the Mono packed fast path: "
				<< error.what() << std::endl;
			return run(false);
		}
	}
	return run(false);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void MonoCudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B, int R_capacity, int B_capacity,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs_dc,
	const float* shs_rest,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
    const float* projmatrix_raw,
    const float* campos,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float center_x, float center_y,
	const int* radii,
	char* geom_buffer,
	char* tile_buffer,
	char* instance_buffer,
	char* image_fixed_buffer,
	char* bucket_buffer,
	char* geometry_plane_buffer,
	char* geometry_bucket_buffer,
	const float* rendered_color,
	const float* rendered_depth,
	const float* rendered_opacity,
	const float* rendered_normal,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	const float* dL_dpix_opacity,
	const float* dL_dpix_normal,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dplanes,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh_dc,
	float* dL_dsh_rest,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dtau,
	float* fisher_dL_dmean3D,
	float* fisher_dL_dopacity,
	bool debug)
{
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const int n_tiles = tile_grid.x * tile_grid.y;

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunks(
		tile_buffer,
		instance_buffer,
		static_cast<size_t>(R_capacity),
		static_cast<size_t>(n_tiles));
	ImageState imgState = ImageState::fromChunks(
		image_fixed_buffer,
		bucket_buffer,
		static_cast<size_t>(width) * static_cast<size_t>(height),
		static_cast<size_t>(n_tiles),
		static_cast<size_t>(B_capacity));

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	float4* plane_ptr = nullptr;
	float4* geometry_bucket_ptr = nullptr;
	if (geometry_plane_buffer != nullptr) {
		char* plane_chunk = geometry_plane_buffer;
		plane_ptr = geometry_planes_from_chunk(plane_chunk, P);
		char* bucket_chunk = geometry_bucket_buffer;
		geometry_bucket_ptr = geometry_buckets_from_chunk(bucket_chunk, B_capacity);
	}

	MONO_CHECK_CUDA(MONO_BACKWARD::render(
		B,
		binningState.tile_ranges,
		binningState.tile_bucket_offsets,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		plane_ptr,
		rendered_color,
		rendered_depth,
		rendered_opacity,
		rendered_normal,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.tile_max_n_processed,
		imgState.bucket_tile_index,
		imgState.bucket_color_transmittance,
		imgState.bucket_depth_accum,
		geometry_bucket_ptr,
		dL_dpix,
		dL_dpix_depth,
		dL_dpix_opacity,
		dL_dpix_normal,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth,
		(float4*)dL_dplanes,
		fisher_dL_dmean3D,
		fisher_dL_dopacity,
		P,
		D,
		M,
		(const float3*)means3D,
		shs_dc,
		shs_rest,
		geomState.clamped,
		(const glm::vec3*)scales,
		(const glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x,
		focal_y,
		center_x,
		center_y,
		tan_fovx,
		tan_fovy,
		(const glm::vec3*)campos
	    ), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const bool need_preprocess = dL_dmean2D != nullptr || dL_dconic != nullptr
		|| dL_dopacity != nullptr || dL_dcolor != nullptr || dL_ddepth != nullptr
		|| dL_dplanes != nullptr
		|| dL_dmean3D != nullptr || dL_dcov3D != nullptr || dL_dsh_dc != nullptr
		|| dL_dsh_rest != nullptr || dL_dscale != nullptr || dL_drot != nullptr
		|| dL_dtau != nullptr;
	if (need_preprocess) {
		MONO_CHECK_CUDA(MONO_BACKWARD::preprocess(P, D, M,
			(float3*)means3D,
			radii,
			shs_dc,
			shs_rest,
			geomState.clamped,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			cov3D_ptr,
			viewmatrix,
			projmatrix,
	        projmatrix_raw,
			focal_x, focal_y,
			tan_fovx, tan_fovy,
			(glm::vec3*)campos,
			(float3*)dL_dmean2D,
			dL_dconic,
			(glm::vec3*)dL_dmean3D,
			dL_dcolor,
			dL_ddepth,
			(float4*)dL_dplanes,
			dL_dcov3D,
			dL_dsh_dc,
			dL_dsh_rest,
			(glm::vec3*)dL_dscale,
			(glm::vec4*)dL_drot,
			dL_dtau), debug)
	}
}


