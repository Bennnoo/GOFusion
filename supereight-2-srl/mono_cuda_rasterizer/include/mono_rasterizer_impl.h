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

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include "mono_rasterizer.h"
#include <cuda_runtime_api.h>

namespace MonoCudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t depth_sorting_size;
		size_t visible_scan_size;
		float* depths;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* tiles_touched;
		uint32_t* visible_depth_keys_unsorted;
		uint32_t* visible_depth_keys;
		uint32_t* visible_point_indices_unsorted;
		uint32_t* visible_point_indices;
		uint32_t* visible_tiles_touched;
		uint32_t* visible_offsets;
		uint32_t* n_visible;
		uint32_t* n_instances;
		char* depth_sorting_space;
		char* visible_scan_space;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t* n_contrib;
		float* accum_alpha;
		uint32_t* tile_max_n_processed;
		uint32_t* bucket_tile_index;
		float4* bucket_color_transmittance;
		float* bucket_depth_accum;

		static ImageState fromChunks(char*& fixed_chunk,
			char*& bucket_chunk,
			size_t N,
			size_t T,
			size_t B);
	};

	struct BinningState
	{
		size_t tile_sorting_size;
		size_t bucket_scan_size;
		uint2* tile_ranges;
		uint32_t* tile_n_buckets;
		uint32_t* tile_bucket_offsets;
		uint32_t* validation_error;
		uint32_t* validation_details;
		int tile_sort_end_bit;
		uint32_t padding_key;
		bool use_u16_tile_keys;
		uint16_t* point_list_keys_u16_unsorted;
		uint16_t* point_list_keys_u16;
		uint32_t* point_list_keys_u32_unsorted;
		uint32_t* point_list_keys_u32;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* tile_sorting_space;
		char* bucket_scan_space;

		static BinningState fromChunks(char*& tile_chunk,
			char*& instance_chunk,
			size_t R,
			size_t T);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	inline size_t required_tile(size_t T)
	{
		char* tile_size = nullptr;
		char* instance_size = nullptr;
		BinningState::fromChunks(tile_size, instance_size, 1, T);
		return reinterpret_cast<size_t>(tile_size) + 128;
	}

	inline size_t required_instance(size_t R, size_t T)
	{
		char* tile_size = nullptr;
		char* instance_size = nullptr;
		BinningState::fromChunks(tile_size, instance_size, R, T);
		return reinterpret_cast<size_t>(instance_size) + 128;
	}

	inline size_t required_image_fixed(size_t N, size_t T)
	{
		char* fixed_size = nullptr;
		char* bucket_size = nullptr;
		ImageState::fromChunks(fixed_size, bucket_size, N, T, 1);
		return reinterpret_cast<size_t>(fixed_size) + 128;
	}

	inline size_t required_bucket(size_t B)
	{
		char* fixed_size = nullptr;
		char* bucket_size = nullptr;
		ImageState::fromChunks(fixed_size, bucket_size, 1, 1, B);
		return reinterpret_cast<size_t>(bucket_size) + 128;
	}

	inline size_t required_geometry_planes(size_t P)
	{
		char* size = nullptr;
		float4* planes = nullptr;
		obtain(size, planes, P, 128);
		return reinterpret_cast<size_t>(size) + 128;
	}

	inline size_t required_geometry_buckets(size_t B)
	{
		char* size = nullptr;
		float4* geometry = nullptr;
		obtain(size, geometry, B * 256, 128);
		return reinterpret_cast<size_t>(size) + 128;
	}

	inline float4* geometry_planes_from_chunk(char*& chunk, size_t P)
	{
		float4* planes = nullptr;
		obtain(chunk, planes, P, 128);
		return planes;
	}

	inline float4* geometry_buckets_from_chunk(char*& chunk, size_t B)
	{
		float4* geometry = nullptr;
		obtain(chunk, geometry, B * 256, 128);
		return geometry;
	}
};
