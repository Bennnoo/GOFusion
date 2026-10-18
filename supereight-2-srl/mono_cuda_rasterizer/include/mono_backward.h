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

#ifndef MONO_CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define MONO_CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace MONO_BACKWARD
{
	void render(
		const int n_buckets,
		const uint2* ranges,
		const uint32_t* tile_bucket_offsets,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float4* planes,
		const float* rendered_color,
		const float* rendered_depth,
		const float* rendered_opacity,
		const float* rendered_normal,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const uint32_t* tile_max_n_processed,
		const uint32_t* bucket_tile_index,
		const float4* bucket_color_transmittance,
		const float* bucket_depth_accum,
		const float4* bucket_geometry_accum,
		const float* dL_dpixels,
		const float* dL_dpixels_depth,
		const float* dL_dpixels_opacity,
		const float* dL_dpixels_normal,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths,
		float4* dL_dplanes,
		float* fisher_dL_dmean3D,
		float* fisher_dL_dopacity,
		int P,
		int D,
		int M,
		const float3* means3D,
		const float* shs_dc,
		const float* shs_rest,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x,
		const float focal_y,
		const float center_x,
		const float center_y,
		const float tan_fovx,
		const float tan_fovy,
		const glm::vec3* campos);

	

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs_dc,
		const float* shs_rest,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_ddepth,
		const float4* dL_dplanes,
		float* dL_dcov3D,
		float* dL_dsh_dc,
		float* dL_dsh_rest,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau);
}

#endif
