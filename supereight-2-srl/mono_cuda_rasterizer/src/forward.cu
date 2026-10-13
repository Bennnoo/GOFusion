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

// Faster-GS portions, modified for GOFusion:
// Copyright 2026 The Faster-GS Authors.
// These portions are licensed under Apache-2.0; see
// supereight-2-srl/fastergs_cuda_rasterizer/LICENSE.

#include "mono_forward.h"
#include "mono_auxiliary.h"
#include "mono_helper_math.h"
#include "mono_math.h"
#include "gs/surfel_geometry.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 mono_computeColorFromSH_fwd(int idx,
	int deg,
	int max_coeffs,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* shs_dc,
	const float* shs_rest,
	bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);
	if (max_coeffs <= 0 || shs_dc == nullptr) {
		clamped[3 * idx + 0] = false;
		clamped[3 * idx + 1] = false;
		clamped[3 * idx + 2] = false;
		return glm::vec3(0.0f);
	}

	const glm::vec3* sh_dc = reinterpret_cast<const glm::vec3*>(shs_dc);
	const int rest_coeffs = max_coeffs - 1;
	const glm::vec3* sh_rest =
		(rest_coeffs > 0 && shs_rest != nullptr)
		? (reinterpret_cast<const glm::vec3*>(shs_rest) + idx * rest_coeffs)
		: nullptr;
	const glm::vec3 sh0 = sh_dc[idx];
	glm::vec3 result = MONO_SH_C0 * sh0;

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - MONO_SH_C1 * y * sh_rest[0] + MONO_SH_C1 * z * sh_rest[1] - MONO_SH_C1 * x * sh_rest[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				MONO_SH_C2[0] * xy * sh_rest[3] +
				MONO_SH_C2[1] * yz * sh_rest[4] +
				MONO_SH_C2[2] * (2.0f * zz - xx - yy) * sh_rest[5] +
				MONO_SH_C2[3] * xz * sh_rest[6] +
				MONO_SH_C2[4] * (xx - yy) * sh_rest[7];

			if (deg > 2)
			{
				result = result +
					MONO_SH_C3[0] * y * (3.0f * xx - yy) * sh_rest[8] +
					MONO_SH_C3[1] * xy * z * sh_rest[9] +
					MONO_SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_rest[10] +
					MONO_SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_rest[11] +
					MONO_SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_rest[12] +
					MONO_SH_C3[5] * z * (xx - yy) * sh_rest[13] +
					MONO_SH_C3[6] * x * (xx - 3.0f * yy) * sh_rest[14];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 mono_computeCov2D_fwd(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = mono_transformPoint4x3_aux(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void mono_computeCov3D_fwd(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	if (!isfinite(scale.x) || !isfinite(scale.y) || !isfinite(scale.z)
		|| !isfinite(mod)
		|| !isfinite(rot.x) || !isfinite(rot.y)
		|| !isfinite(rot.z) || !isfinite(rot.w)) {
		for (int i = 0; i < 6; ++i) {
			cov3D[i] = 0.0f;
		}
		return;
	}

	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	const float q_norm2 = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w;
	if (!isfinite(q_norm2) || q_norm2 <= MONO_MIN_QUAT_NORM2) {
		for (int i = 0; i < 6; ++i) {
			cov3D[i] = 0.0f;
		}
		return;
	}
	const float inv_q_norm = rsqrtf(q_norm2);
	glm::vec4 q = rot * inv_q_norm;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void mono_preprocessCUDA_fwd(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs_dc,
	const float* shs_rest,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float4* planes,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!mono_in_frustum_aux(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = mono_transformPoint4x4_aux(p_orig, projmatrix);
	if (!isfinite(p_hom.x) || !isfinite(p_hom.y)
		|| !isfinite(p_hom.z) || !isfinite(p_hom.w)
		|| fabsf(p_hom.w) <= 1.0e-7f) {
		return;
	}
	float p_w = 1.0f / p_hom.w;
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	if (!isfinite(p_proj.x) || !isfinite(p_proj.y) || !isfinite(p_proj.z)) {
		return;
	}

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		mono_computeCov3D_fwd(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
	for (int i = 0; i < 6; ++i) {
		if (!isfinite(cov3D[i])) {
			return;
		}
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = mono_computeCov2D_fwd(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	if (!isfinite(cov.x) || !isfinite(cov.y) || !isfinite(cov.z)
		|| cov.x <= 0.0f || cov.z <= 0.0f) {
		return;
	}

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (!isfinite(det) || det <= MONO_MIN_COV2D_DETERMINANT)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
	if (!isfinite(conic.x) || !isfinite(conic.y) || !isfinite(conic.z)) {
		return;
	}

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { mono_ndc2Pix_aux(p_proj.x, W), mono_ndc2Pix_aux(p_proj.y, H) };
	float4 con_o = { conic.x, conic.y, conic.z, opacities[idx] };
	if (!isfinite(mid) || !isfinite(lambda1) || !isfinite(lambda2)
		|| !isfinite(my_radius) || my_radius <= 0.0f || my_radius > 1.0e9f
		|| !isfinite(point_image.x) || !isfinite(point_image.y)
		|| !isfinite(con_o.w)) {
		return;
	}
	uint32_t tiles_count = mono_duplicateToTilesTouched_aux(
		point_image, con_o, grid,
		0, 0, 0.0f,
		nullptr, nullptr);

	if (tiles_count == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = mono_computeColorFromSH_fwd(
			idx,
			D,
			M,
			(glm::vec3*)orig_points,
			*cam_pos,
			shs_dc,
			shs_rest,
			clamped);
		if (!isfinite(result.x) || !isfinite(result.y) || !isfinite(result.z)) {
			return;
		}
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	else
	{
		for (int channel = 0; channel < C; ++channel) {
			if (!isfinite(colors_precomp[idx * C + channel])) {
				return;
			}
		}
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	if (planes != nullptr) {
		if (rotations == nullptr) {
			planes[idx] = make_float4(0.0f, 0.0f, -1.0f, -p_view.z);
		}
		else {
			const glm::vec4 rotation = rotations[idx];
			bool valid_normal = false;
			const float3 normal_world = gs::surfel::quaternionNormal(
				make_float4(rotation.x, rotation.y, rotation.z, rotation.w),
				&valid_normal);
			float3 normal_camera = mono_transformVec4x3_aux(normal_world, viewmatrix);
			if (!valid_normal || !isfinite(normal_camera.x) || !isfinite(normal_camera.y)
				|| !isfinite(normal_camera.z)) {
				normal_camera = make_float3(0.0f, 0.0f, -1.0f);
			}
			if (normal_camera.x * p_view.x + normal_camera.y * p_view.y
				+ normal_camera.z * p_view.z > 0.0f) {
				normal_camera.x = -normal_camera.x;
				normal_camera.y = -normal_camera.y;
				normal_camera.z = -normal_camera.z;
			}
			planes[idx] = make_float4(
				normal_camera.x,
				normal_camera.y,
				normal_camera.z,
				normal_camera.x * p_view.x + normal_camera.y * p_view.y
					+ normal_camera.z * p_view.z);
		}
	}
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = con_o;
	tiles_touched[idx] = tiles_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool RENDER_GEOMETRY>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
mono_renderCUDA_fwd(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ tile_bucket_offsets,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ tile_max_n_processed,
	uint32_t* __restrict__ bucket_tile_index,
	float4* __restrict__ bucket_color_transmittance,
	float* __restrict__ bucket_depth_accum,
	float4* __restrict__ bucket_geometry_accum,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	const float4* __restrict__ planes,
	float* __restrict__ out_depth, 
	float* __restrict__ out_opacity,
	float* __restrict__ out_normal,
	const float focal_x,
	const float focal_y,
	const float center_x,
	const float center_y,
	int * __restrict__ n_touched,
	const uint32_t* __restrict__ guard_error_flags)
{
	if (guard_error_flags != nullptr && *guard_error_flags != 0U) {
		return;
	}
	auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t tile_idx = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	const uint32_t thread_rank = block.thread_rank();

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[tile_idx];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	const int n_buckets = (toDo + 31) / 32;
	const uint32_t tile_bucket_base = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float4 collected_plane[RENDER_GEOMETRY ? BLOCK_SIZE : 1];

	for (int buckets_remaining = n_buckets, current_bucket = static_cast<int>(thread_rank);
		 buckets_remaining > 0;
		 buckets_remaining -= BLOCK_SIZE, current_bucket += BLOCK_SIZE) {
		if (current_bucket < n_buckets) {
			bucket_tile_index[tile_bucket_base + current_bucket] = tile_idx;
		}
	}

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;
	float3 N = make_float3(0.0f, 0.0f, 0.0f);
	uint32_t n_processed = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
				if constexpr (RENDER_GEOMETRY) {
					collected_plane[block.thread_rank()] = planes[coll_id];
				}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if ((n_processed & 31U) == 0U) {
				const uint32_t bucket_slot = tile_bucket_base + (n_processed >> 5);
				bucket_color_transmittance[bucket_slot * BLOCK_SIZE + thread_rank] =
					make_float4(C[0], C[1], C[2], T);
				bucket_depth_accum[bucket_slot * BLOCK_SIZE + thread_rank] = D;
					if constexpr (RENDER_GEOMETRY) {
						bucket_geometry_accum[bucket_slot * BLOCK_SIZE + thread_rank] =
						make_float4(D, N.x, N.y, N.z);
				}
			}
			n_processed++;

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			float alpha = con_o.w * exp(power);
			if (alpha < MONO_ALPHA_THRESHOLD) {
				continue;
			}

			const float prev_T = T;
			const float blend_weight = alpha * prev_T;
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * blend_weight;
			}
				if constexpr (RENDER_GEOMETRY) {
				const float4 plane = collected_plane[j];
				const float3 ray = make_float3(
					(static_cast<float>(pix.x) + 0.5f - center_x) / focal_x,
					(static_cast<float>(pix.y) + 0.5f - center_y) / focal_y,
					1.0f);
				D += gs::surfel::rayPlaneDepth(plane, ray) * blend_weight;
				N.x += plane.x * blend_weight;
				N.y += plane.y * blend_weight;
				N.z += plane.z * blend_weight;
			}
			else {
				D += collected_depth[j] * blend_weight;
			}
			const float test_T = prev_T * (1.0f - alpha);
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			if (T < MONO_TRANSMITTANCE_THRESHOLD) {
				done = true;
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
			if constexpr (RENDER_GEOMETRY) {
			out_normal[pix_id] = N.x;
			out_normal[H * W + pix_id] = N.y;
			out_normal[2 * H * W + pix_id] = N.z;
		}
	}

	__shared__ uint32_t tile_max_processed_shared[BLOCK_SIZE];
	tile_max_processed_shared[thread_rank] = last_contributor;
	block.sync();
	for (uint32_t stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
		if (thread_rank < stride) {
			tile_max_processed_shared[thread_rank] =
				max(tile_max_processed_shared[thread_rank],
					tile_max_processed_shared[thread_rank + stride]);
		}
		block.sync();
	}
	if (thread_rank == 0) {
		tile_max_n_processed[tile_idx] = tile_max_processed_shared[0];
	}
}

void MONO_FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* tile_bucket_offsets,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* tile_max_n_processed,
	uint32_t* bucket_tile_index,
	float4* bucket_color_transmittance,
	float* bucket_depth_accum,
	float4* bucket_geometry_accum,
	const float* bg_color,
	float* out_color,
	const float* depth,
	const float4* planes,
	float* out_depth, 
	float* out_opacity,
	float* out_normal,
	float focal_x, float focal_y,
	float center_x, float center_y,
	bool render_geometry,
	int* n_touched,
	const uint32_t* guard_error_flags,
	cudaStream_t stream)
{
#define MONO_RENDER_FORWARD_ARGS                                                \
		ranges,                                                                    \
		tile_bucket_offsets,                                                       \
		point_list,                                                                \
		W, H,                                                                      \
		means2D,                                                                   \
		colors,                                                                    \
		conic_opacity,                                                             \
		final_T,                                                                   \
		n_contrib,                                                                 \
		tile_max_n_processed,                                                      \
		bucket_tile_index,                                                         \
		bucket_color_transmittance,                                                \
		bucket_depth_accum,                                                        \
		bucket_geometry_accum,                                                     \
		bg_color,                                                                  \
		out_color,                                                                 \
		depth,                                                                     \
		planes,                                                                    \
		out_depth,                                                                 \
		out_opacity,                                                               \
		out_normal,                                                                \
		focal_x,                                                                   \
		focal_y,                                                                   \
		center_x,                                                                  \
		center_y,                                                                  \
		n_touched,                                                                 \
		guard_error_flags
	if (render_geometry) {
		mono_renderCUDA_fwd<NUM_CHANNELS, true><<<grid, block, 0, stream>>>(
			MONO_RENDER_FORWARD_ARGS);
	}
	else {
		mono_renderCUDA_fwd<NUM_CHANNELS, false><<<grid, block, 0, stream>>>(
			MONO_RENDER_FORWARD_ARGS);
	}
#undef MONO_RENDER_FORWARD_ARGS
}

void MONO_FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs_dc,
	const float* shs_rest,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float4* planes,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	cudaStream_t stream)
{
	mono_preprocessCUDA_fwd<NUM_CHANNELS> << <(P + 255) / 256, 256, 0, stream >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs_dc,
		shs_rest,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		planes,
		grid,
		tiles_touched,
		prefiltered
		);
}
