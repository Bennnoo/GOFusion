/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace gs::detail {

struct alignas(32) RasterCountSummary {
    uint32_t n_visible = 0;
    uint32_t n_instances = 0;
    uint32_t n_buckets = 0;
    uint32_t error_flags = 0;
    uint32_t required_instance_capacity = 0;
    uint32_t required_bucket_capacity = 0;
    uint32_t detail0 = 0;
    uint32_t detail1 = 0;
};

static_assert(sizeof(RasterCountSummary) == 32,
              "RasterCountSummary must remain a single 32-byte packet");

inline constexpr uint32_t kRasterOverflowInstances = 1u << 28;
inline constexpr uint32_t kRasterOverflowBuckets = 1u << 29;
inline constexpr uint32_t kRasterInvalidCounts = 1u << 30;

struct RasterDependencyStats {
    std::atomic<uint64_t> packed_summary_sync_count{0};
    std::atomic<uint64_t> raster_capacity_grows{0};
    std::atomic<uint64_t> raster_retries{0};
    std::atomic<uint64_t> overflow_required_bytes{0};
    std::atomic<uint64_t> instance_high_water{0};
    std::atomic<uint64_t> bucket_high_water{0};
    std::atomic<uint64_t> mono_exact_fallbacks{0};
    std::atomic<uint64_t> invalid_summaries{0};
    std::atomic<uint64_t> bucket_only_grows{0};
    std::atomic<uint64_t> scratch_reserved_bytes{0};
    std::atomic<uint64_t> scratch_peak_reserved_bytes{0};
    std::atomic<uint64_t> scratch_slot_count{0};
};

inline RasterDependencyStats& rasterDependencyStats()
{
    static RasterDependencyStats stats;
    return stats;
}

inline void updateAtomicMax(std::atomic<uint64_t>& target, const uint64_t value)
{
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value
           && !target.compare_exchange_weak(current,
                                            value,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
    }
}

inline bool environmentFlagEnabled(const char* name, const bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
        && std::strcmp(value, "FALSE") != 0;
}

inline std::atomic<bool>& monoPackedRasterRuntimeState()
{
    static std::atomic<bool> enabled{true};
    return enabled;
}

inline bool packedMonoRasterSummaryEnabled()
{
    static const bool configured = environmentFlagEnabled(
        "GOFUSION_MONO_RASTER_PACKED_SUMMARY",
        environmentFlagEnabled("GOFUSION_RASTER_PACKED_SUMMARY", true));
    return configured && monoPackedRasterRuntimeState().load(std::memory_order_relaxed);
}

inline bool packedFasterRasterSummaryEnabled()
{
    static const bool enabled = environmentFlagEnabled(
        "GOFUSION_FASTER_RASTER_PACKED_SUMMARY",
        environmentFlagEnabled("GOFUSION_RASTER_PACKED_SUMMARY", true));
    return enabled;
}

inline bool packedRasterSummaryEnabled()
{
    return packedFasterRasterSummaryEnabled();
}

inline void disableMonoPackedRasterRuntime()
{
    monoPackedRasterRuntimeState().store(false, std::memory_order_relaxed);
}

inline bool rasterDependencyStatsEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GOFUSION_RASTER_STATS");
        return value != nullptr && std::strcmp(value, "0") != 0
            && std::strcmp(value, "false") != 0
            && std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

inline void maybeReportRasterDependencyStats(const char* backend,
                                             const RasterCountSummary& summary,
                                             const uint32_t instance_capacity,
                                             const uint32_t bucket_capacity)
{
    if (!rasterDependencyStatsEnabled()) return;
    const auto& stats = rasterDependencyStats();
    const uint64_t syncs =
        stats.packed_summary_sync_count.load(std::memory_order_relaxed);
    const bool overflow = (summary.error_flags
                           & (kRasterOverflowInstances | kRasterOverflowBuckets))
        != 0;

    if (!overflow && syncs > 5u && syncs % 1000u != 0) return;
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << "Raster dependency stats [" << backend << "]: syncs=" << syncs
              << " grows="
              << stats.raster_capacity_grows.load(std::memory_order_relaxed)
              << " retries=" << stats.raster_retries.load(std::memory_order_relaxed)
              << " instances=" << summary.n_instances << "/" << instance_capacity
              << " buckets=" << summary.n_buckets << "/" << bucket_capacity
              << " high_water="
              << stats.instance_high_water.load(std::memory_order_relaxed) << "/"
              << stats.bucket_high_water.load(std::memory_order_relaxed)
              << " overflow_required_bytes="
              << stats.overflow_required_bytes.load(std::memory_order_relaxed)
              << " exact_fallbacks="
              << stats.mono_exact_fallbacks.load(std::memory_order_relaxed)
              << " invalid_summaries="
              << stats.invalid_summaries.load(std::memory_order_relaxed)
              << " bucket_only_grows="
              << stats.bucket_only_grows.load(std::memory_order_relaxed)
              << " scratch_reserved_bytes="
              << stats.scratch_reserved_bytes.load(std::memory_order_relaxed)
              << "/"
              << stats.scratch_peak_reserved_bytes.load(std::memory_order_relaxed)
              << " slots="
              << stats.scratch_slot_count.load(std::memory_order_relaxed)
              << std::endl;
}

inline size_t checkedRasterBytes(const uint64_t elements, const uint64_t bytes_per_element)
{
    if (bytes_per_element != 0
        && elements > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
                / bytes_per_element) {
        throw std::overflow_error("raster workspace byte-size overflow");
    }
    return static_cast<size_t>(elements * bytes_per_element);
}

} 
