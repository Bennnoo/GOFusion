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

#ifndef MONO_CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define MONO_CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "mono_config.h"
#include "stdio.h"
#include <math.h>
#include <stdint.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float MONO_SH_C0 = 0.28209479177387814f;
__device__ const float MONO_SH_C1 = 0.4886025119029199f;
__device__ const float MONO_SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float MONO_SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float mono_ndc2Pix_aux(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void mono_getRect_aux(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 mono_transformPoint4x3_aux(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 mono_transformPoint4x4_aux(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 mono_transformVec4x3_aux(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 mono_transformVec4x3Transpose_aux(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float mono_dnormvdz_aux(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float mono_dnormvdz_aux = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return mono_dnormvdz_aux;
}

__forceinline__ __device__ float3 mono_dnormvdv_aux(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 mono_dnormvdv_aux;
	mono_dnormvdv_aux.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	mono_dnormvdv_aux.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	mono_dnormvdv_aux.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return mono_dnormvdv_aux;
}

__forceinline__ __device__ float4 mono_dnormvdv_aux(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 mono_dnormvdv_aux;
	mono_dnormvdv_aux.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	mono_dnormvdv_aux.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	mono_dnormvdv_aux.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	mono_dnormvdv_aux.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return mono_dnormvdv_aux;
}

__forceinline__ __device__ float mono_sigmoid_aux(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool mono_in_frustum_aux(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	(void) projmatrix;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	if (!isfinite(p_orig.x) || !isfinite(p_orig.y) || !isfinite(p_orig.z)) {
		return false;
	}

	// Bring points to camera space.
	p_view = mono_transformPoint4x3_aux(p_orig, viewmatrix);

	if (!isfinite(p_view.x) || !isfinite(p_view.y) || !isfinite(p_view.z)
		|| p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__device__ inline bool mono_computeEllipseIntersection_aux(
	const float4 con_o, const float disc, const float t, const float2 p,
	const bool isY, const float coord, float2* intersection)
{
	const float p_u = isY ? p.y : p.x;
	const float p_v = isY ? p.x : p.y;
	const float coeff = isY ? con_o.x : con_o.z;
	const float h = coord - p_u;
	const float radicand = disc * h * h + t * coeff;
	const float tolerance = 1.0e-5f * fmaxf(1.0f, fabsf(t * coeff));
	if (!isfinite(radicand) || radicand < -tolerance) {
		return false;
	}

	const float sqrt_term = sqrtf(fmaxf(radicand, 0.0f));
	const float2 value = {
		(-con_o.y * h - sqrt_term) / coeff + p_v,
		(-con_o.y * h + sqrt_term) / coeff + p_v};
	if (!isfinite(value.x) || !isfinite(value.y)) {
		return false;
	}
	*intersection = value;
	return true;
}

__forceinline__ __device__ bool mono_validConicOpacity_aux(
	const float2 p, const float4 con_o, float& disc)
{
	if (!isfinite(p.x) || !isfinite(p.y)
		|| !isfinite(con_o.x) || !isfinite(con_o.y) || !isfinite(con_o.z)
		|| !isfinite(con_o.w) || con_o.w <= MONO_ALPHA_THRESHOLD) {
		return false;
	}

	disc = con_o.y * con_o.y - con_o.x * con_o.z;
	return isfinite(disc) && con_o.x > 0.0f && con_o.z > 0.0f && disc < 0.0f;
}

__forceinline__ __device__ int mono_tileLowerBound_aux(
	const float coord, const float block_size, const int lower, const int upper)
{
	const float scaled = fminf(
		static_cast<float>(upper),
		fmaxf(static_cast<float>(lower), coord / block_size));
	return __float2int_rd(scaled);
}

__forceinline__ __device__ int mono_tileUpperBound_aux(
	const float coord, const float block_size, const int lower, const int upper)
{
	const float scaled = fminf(
		static_cast<float>(upper),
		fmaxf(static_cast<float>(lower - 1), coord / block_size));
	return min(upper, __float2int_rd(scaled) + 1);
}

template <typename KeyT>
__device__ inline uint32_t mono_processTilesByTileKey_aux(
	const float4 con_o, const float disc, const float t, const float2 p,
	float2 bbox_min, float2 bbox_max,
	float2 bbox_argmin, float2 bbox_argmax,
	int2 rect_min, int2 rect_max,
	const dim3 grid, const bool isY,
	const uint32_t idx, const uint32_t off, const uint32_t max_writes,
	KeyT* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted)
{
	const float block_u = isY ? static_cast<float>(BLOCK_Y) : static_cast<float>(BLOCK_X);
	const float block_v = isY ? static_cast<float>(BLOCK_X) : static_cast<float>(BLOCK_Y);

	if (isY) {
		rect_min = {rect_min.y, rect_min.x};
		rect_max = {rect_max.y, rect_max.x};
		bbox_min = {bbox_min.y, bbox_min.x};
		bbox_max = {bbox_max.y, bbox_max.x};
		bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
		bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
	}

	uint32_t tiles_count = 0;
	float2 intersect_max_line = {bbox_max.y, bbox_min.y};
	float2 intersect_min_line = intersect_max_line;
	bool intersect_min_valid = true;
	bool intersect_max_valid = true;

	float min_line = rect_min.x * block_u;
	if (bbox_min.x <= min_line) {
		intersect_min_valid = mono_computeEllipseIntersection_aux(
			con_o, disc, t, p, isY, min_line, &intersect_min_line);
	}

	for (int u = rect_min.x; u < rect_max.x; ++u) {
		const float max_line = min_line + block_u;
		if (max_line <= bbox_max.x) {
			intersect_max_valid = mono_computeEllipseIntersection_aux(
				con_o, disc, t, p, isY, max_line, &intersect_max_line);
		}

		float ellipse_min = 0.0f;
		float ellipse_max = 0.0f;
		bool stripe_valid = true;
		if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
			ellipse_min = bbox_min.y;
		} else if (intersect_min_valid && intersect_max_valid) {
			ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
		} else {
			stripe_valid = false;
		}

		if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
			ellipse_max = bbox_max.y;
		} else if (intersect_min_valid && intersect_max_valid) {
			ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
		} else {
			stripe_valid = false;
		}

		if (stripe_valid && isfinite(ellipse_min) && isfinite(ellipse_max)) {
			const int min_tile_v = mono_tileLowerBound_aux(
				ellipse_min, block_v, rect_min.y, rect_max.y);
			const int max_tile_v = mono_tileUpperBound_aux(
				ellipse_max, block_v, rect_min.y, rect_max.y);
			const int tile_span = max_tile_v - min_tile_v;

			// Keep the subtraction signed. Casting a negative span to uint32_t
			// was the source of the 2^32 instance-count underflow.
			if (tile_span > 0) {
				const uint32_t span = static_cast<uint32_t>(tile_span);
				if (gaussian_keys_unsorted != nullptr) {
					for (int v = min_tile_v; v < max_tile_v; ++v) {
						const uint32_t local_offset =
							tiles_count + static_cast<uint32_t>(v - min_tile_v);
						if (local_offset >= max_writes) {
							continue;
						}
						const uint32_t tile_idx =
							isY ? static_cast<uint32_t>(u * grid.x + v)
							    : static_cast<uint32_t>(v * grid.x + u);
						gaussian_keys_unsorted[off + local_offset] =
							static_cast<KeyT>(tile_idx);
						gaussian_values_unsorted[off + local_offset] = idx;
					}
				}
				tiles_count += span;
			}
		}

		intersect_min_line = intersect_max_line;
		intersect_min_valid = intersect_max_valid;
		min_line = max_line;
	}

	return tiles_count;
}

template <typename KeyT>
__device__ inline uint32_t mono_duplicateToTilesTouchedByTileKey_aux(
	const float2 p, const float4 con_o, const dim3 grid, const uint32_t idx,
	const uint32_t off, const uint32_t max_writes,
	KeyT* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted)
{
	float disc = 0.0f;
	if (!mono_validConicOpacity_aux(p, con_o, disc) || grid.x == 0 || grid.y == 0) {
		return 0;
	}

	const float t = 2.0f * logf(con_o.w * 255.0f);
	const float x_term_sq = -(con_o.y * con_o.y * t) / (disc * con_o.x);
	const float y_term_sq = -(con_o.y * con_o.y * t) / (disc * con_o.z);
	if (!isfinite(t) || t < 0.0f
		|| !isfinite(x_term_sq) || x_term_sq < 0.0f
		|| !isfinite(y_term_sq) || y_term_sq < 0.0f) {
		return 0;
	}

	float x_term = sqrtf(x_term_sq);
	x_term = (con_o.y < 0.0f) ? x_term : -x_term;
	float y_term = sqrtf(y_term_sq);
	y_term = (con_o.y < 0.0f) ? y_term : -y_term;

	const float2 bbox_argmin = {p.y - y_term, p.x - x_term};
	const float2 bbox_argmax = {p.y + y_term, p.x + x_term};
	if (!isfinite(bbox_argmin.x) || !isfinite(bbox_argmin.y)
		|| !isfinite(bbox_argmax.x) || !isfinite(bbox_argmax.y)) {
		return 0;
	}

	float2 bbox_min_y;
	float2 bbox_min_x;
	float2 bbox_max_y;
	float2 bbox_max_x;
	if (!mono_computeEllipseIntersection_aux(
			con_o, disc, t, p, true, bbox_argmin.x, &bbox_min_y)
		|| !mono_computeEllipseIntersection_aux(
			con_o, disc, t, p, false, bbox_argmin.y, &bbox_min_x)
		|| !mono_computeEllipseIntersection_aux(
			con_o, disc, t, p, true, bbox_argmax.x, &bbox_max_y)
		|| !mono_computeEllipseIntersection_aux(
			con_o, disc, t, p, false, bbox_argmax.y, &bbox_max_x)) {
		return 0;
	}

	const float2 bbox_min = {bbox_min_y.x, bbox_min_x.x};
	const float2 bbox_max = {bbox_max_y.y, bbox_max_x.y};
	if (!isfinite(bbox_min.x) || !isfinite(bbox_min.y)
		|| !isfinite(bbox_max.x) || !isfinite(bbox_max.y)
		|| bbox_min.x > bbox_max.x || bbox_min.y > bbox_max.y) {
		return 0;
	}

	const int2 rect_min = {
		mono_tileLowerBound_aux(
			bbox_min.x, static_cast<float>(BLOCK_X), 0, static_cast<int>(grid.x)),
		mono_tileLowerBound_aux(
			bbox_min.y, static_cast<float>(BLOCK_Y), 0, static_cast<int>(grid.y))};
	const int2 rect_max = {
		mono_tileUpperBound_aux(
			bbox_max.x, static_cast<float>(BLOCK_X), 0, static_cast<int>(grid.x)),
		mono_tileUpperBound_aux(
			bbox_max.y, static_cast<float>(BLOCK_Y), 0, static_cast<int>(grid.y))};

	const int x_span = rect_max.x - rect_min.x;
	const int y_span = rect_max.y - rect_min.y;
	if (x_span <= 0 || y_span <= 0) {
		return 0;
	}

	return mono_processTilesByTileKey_aux<KeyT>(
		con_o,
		disc,
		t,
		p,
		bbox_min,
		bbox_max,
		bbox_argmin,
		bbox_argmax,
		rect_min,
		rect_max,
		grid,
		y_span < x_span,
		idx,
		off,
		max_writes,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
}

__device__ inline uint32_t mono_duplicateToTilesTouched_aux(
	const float2 p, const float4 con_o, const dim3 grid, uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted)
{
	(void) depth;
	(void) gaussian_keys_unsorted;
	return mono_duplicateToTilesTouchedByTileKey_aux<uint32_t>(
		p,
		con_o,
		grid,
		idx,
		off,
		UINT32_MAX,
		nullptr,
		gaussian_values_unsorted);
}

#define MONO_CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
