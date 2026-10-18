// Derived from Faster-GS, Copyright 2026 The Faster-GS Authors.
// SPDX-License-Identifier: Apache-2.0
// Modified for GOFusion; differs from the upstream Faster-GS implementation.
// See LICENSE in this directory tree.
// SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

#pragma once

#include "fastergs/rasterization_config.h"
#include "fastergs/helper_math.h"
#include <cub/cub.cuh>
#include <cstdint>

namespace faster_gs::rasterization {

    inline __host__ int extract_end_bit(uint n) {
        int leading_zeros = 0;
        if ((n & 0xffff0000) == 0) { leading_zeros += 16; n <<= 16; }
        if ((n & 0xff000000) == 0) { leading_zeros += 8; n <<= 8; }
        if ((n & 0xf0000000) == 0) { leading_zeros += 4; n <<= 4; }
        if ((n & 0xc0000000) == 0) { leading_zeros += 2; n <<= 2; }
        if ((n & 0x80000000) == 0) { leading_zeros += 1; }
        return 32 - leading_zeros;
    }

    inline __host__ int tile_key_end_bit(uint n_tiles) {
        return n_tiles > 1 ? extract_end_bit(n_tiles - 1) : 1;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    template <typename T>
    static void obtain(char*& blob, T*& ptr, size_t count) {
        constexpr size_t alignment = 128;
        size_t offset = (reinterpret_cast<size_t>(blob) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template <typename T, typename... Args>
    size_t required(size_t P, Args... args) {
        char* size = nullptr;
        T::from_blob(size, P, args...);
        return ((size_t)size);
    }

    struct PrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<uint> depth_keys;
        cub::DoubleBuffer<uint> primitive_indices;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;
        float3* color;
        uint* n_visible_primitives;
        uint* n_instances;

        static PrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            PrimitiveBuffers buffers;
            uint* depth_keys_current;
            obtain(blob, depth_keys_current, n_primitives);
            uint* depth_keys_alternate;
            obtain(blob, depth_keys_alternate, n_primitives);
            buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_primitives);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_primitives);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            obtain(blob, buffers.n_touched_tiles, n_primitives);
            obtain(blob, buffers.offset, n_primitives);
            obtain(blob, buffers.screen_bounds, n_primitives);
            obtain(blob, buffers.mean2d, n_primitives);
            obtain(blob, buffers.conic_opacity, n_primitives);
            obtain(blob, buffers.color, n_primitives);
            cub::DeviceScan::ExclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.offset, buffers.offset,
                n_primitives
            );
            size_t sorting_workspace_size;
            cub::DeviceRadixSort::SortPairs(
                nullptr, sorting_workspace_size,
                buffers.depth_keys, buffers.primitive_indices,
                n_primitives
            );
            buffers.cub_workspace_size = max(buffers.cub_workspace_size, sorting_workspace_size);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size);
            obtain(blob, buffers.n_visible_primitives, 1);
            obtain(blob, buffers.n_instances, 1);
            return buffers;
        }
    };

    template <typename KeyT>
    struct InstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<KeyT> keys;
        cub::DoubleBuffer<uint> primitive_indices;

        static InstanceBuffers from_blob(char*& blob, int n_instances, int end_bit) {
            InstanceBuffers buffers;
            KeyT* keys_current;
            obtain(blob, keys_current, n_instances);
            KeyT* keys_alternate;
            obtain(blob, keys_alternate, n_instances);
            buffers.keys = cub::DoubleBuffer<KeyT>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances,
                0, end_bit
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size);
            return buffers;
        }
    };

    struct TileBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint2* instance_ranges;
        float* final_transmittances;
        uint* n_buckets;
        uint* buckets_offset;
        uint* validation_error;
        uint* max_n_processed;
        uint* n_processed;

        static TileBuffers from_blob(char*& blob, int n_tiles) {
            TileBuffers buffers;
            const size_t tile_count = static_cast<size_t>(n_tiles);
            obtain(blob, buffers.instance_ranges, tile_count);
            obtain(blob, buffers.n_buckets, tile_count);
            obtain(blob, buffers.buckets_offset, tile_count);
            obtain(blob, buffers.validation_error, 1);
            obtain(blob, buffers.max_n_processed, tile_count);
            obtain(blob,
                   buffers.n_processed,
                   tile_count * static_cast<size_t>(config::block_size_blend));
            obtain(blob,
                   buffers.final_transmittances,
                   tile_count * static_cast<size_t>(config::block_size_blend));
            cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_buckets, buffers.buckets_offset,
                n_tiles
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size);
            return buffers;
        }
    };

    struct BucketBuffers {
        uint* tile_index;
        float4* color_transmittance;

        static BucketBuffers from_blob(char*& blob, int n_buckets) {
            BucketBuffers buffers;
            const size_t bucket_entries = static_cast<size_t>(n_buckets)
                * static_cast<size_t>(config::block_size_blend);
            obtain(blob, buffers.tile_index, bucket_entries);
            obtain(blob, buffers.color_transmittance, bucket_entries);
            return buffers;
        }
    };

    // Kept outside the legacy packed buffers so RGB-only renders retain the
    // exact allocation footprint and scratch-cache semantics.
    struct GeometryPrimitiveBuffers {
        float4* planes;

        static GeometryPrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            GeometryPrimitiveBuffers buffers;
            obtain(blob, buffers.planes, n_primitives);
            return buffers;
        }
    };

    struct GeometryBucketBuffers {
        float4* depth_normal;

        static GeometryBucketBuffers from_blob(char*& blob, int n_buckets) {
            GeometryBucketBuffers buffers;
            obtain(blob,
                   buffers.depth_normal,
                   static_cast<size_t>(n_buckets)
                       * static_cast<size_t>(config::block_size_blend));
            return buffers;
        }
    };


}
