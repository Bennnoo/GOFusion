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

#ifndef MONO_CUDA_RASTERIZER_H_INCLUDED
#define MONO_CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "gs/raster_count_summary.cuh"

namespace MonoCudaRasterizer
{
	class Rasterizer
	{
	public:

		static int forward(
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
			int* radii = nullptr,
			int* n_touched = nullptr,
			int* n_buckets = nullptr,
			int* instance_capacity = nullptr,
			int* bucket_capacity = nullptr,
			bool debug = false);

		static void backward(
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
			float* dL_ddepths,
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
			bool debug);

		
	};
};

#endif
