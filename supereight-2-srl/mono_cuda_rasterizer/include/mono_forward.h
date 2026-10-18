/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#ifndef MONO_CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define MONO_CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace MONO_FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		float4* planes,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		cudaStream_t stream = nullptr);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* tile_bucket_offsets,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
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
		const uint32_t* guard_error_flags = nullptr,
		cudaStream_t stream = nullptr);
}


#endif
