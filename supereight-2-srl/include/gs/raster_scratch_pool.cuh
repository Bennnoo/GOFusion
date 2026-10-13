/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "gs/raster_count_summary.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <utility>

namespace gs::detail {

template<size_t BufferCount, size_t LogicalCapacityCount = 2>
class RasterScratchPool {
    private:
    struct Slot {
        explicit Slot(const int device_index_in) : device_index(device_index_in)
        {
            const auto options =
                torch::TensorOptions().dtype(torch::kUInt8).device(
                    torch::Device(torch::kCUDA, device_index));
            for (auto& buffer : buffers) {
                buffer = torch::empty({0}, options);
            }
            summary_device = torch::empty(
                {static_cast<int64_t>(sizeof(RasterCountSummary))}, options);
            summary_host = torch::empty(
                {static_cast<int64_t>(sizeof(RasterCountSummary))},
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(torch::kCPU)
                    .pinned_memory(true));
        }

        int device_index = -1;
        bool in_use = false;
        std::array<torch::Tensor, BufferCount> buffers;
        std::array<int64_t, BufferCount> capacities{};
        std::array<uint32_t, LogicalCapacityCount> logical_capacities{};
        torch::Tensor summary_device;
        torch::Tensor summary_host;
        at::cuda::CUDAEvent ready_event;
        at::cuda::CUDAEvent summary_ready_event;
    };

    public:
    static constexpr int64_t kInvalidSlot = -1;

    struct Lease {
        int64_t id = kInvalidSlot;
        std::array<torch::Tensor, BufferCount> buffers;
        std::array<std::function<char*(size_t)>, BufferCount> resize;
        std::array<uint32_t, LogicalCapacityCount> logical_capacities{};
        std::array<std::function<uint32_t(uint32_t)>, LogicalCapacityCount>
            reserve_logical;
        RasterCountSummary* summary_device = nullptr;
        std::function<const RasterCountSummary&()> read_summary;
    };

    Lease acquire(const int device_index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto stream = c10::cuda::getCurrentCUDAStream(device_index);

        Slot* selected = nullptr;
        int64_t selected_id = kInvalidSlot;
        for (size_t i = 0; i < slots_.size(); ++i) {
            auto& slot = slots_[i];
            if (slot.device_index != device_index) {
                continue;
            }

            if (slot.in_use && exclusivelyOwnedByPool(slot)) {
                slot.in_use = false;
            }
            if (!slot.in_use && exclusivelyOwnedByPool(slot)) {
                selected = &slot;
                selected_id = static_cast<int64_t>(i);
                break;
            }
        }

        if (!selected) {
            slots_.emplace_back(device_index);
            rasterDependencyStats().scratch_slot_count.fetch_add(
                1, std::memory_order_relaxed);
            selected = &slots_.back();
            selected_id = static_cast<int64_t>(slots_.size() - 1);
        }

        selected->ready_event.block(stream);
        selected->in_use = true;

        Lease lease;
        lease.id = selected_id;
        for (size_t i = 0; i < BufferCount; ++i) {
            lease.buffers[i] = selected->buffers[i];
            lease.resize[i] = [selected, i](const size_t required_bytes) {
                const int64_t required = static_cast<int64_t>(required_bytes);
                auto& capacity = selected->capacities[i];
                auto& buffer = selected->buffers[i];
                if (required > capacity) {
                    const int64_t minimum_growth = 4096;
                    const int64_t proportional_growth = capacity / 4;
                    const int64_t grown =
                        capacity + std::max(minimum_growth, proportional_growth);
                    int64_t target = std::max(required, grown);
                    constexpr int64_t alignment = 256;
                    target = ((target + alignment - 1) / alignment) * alignment;
                    buffer.resize_({target});
                    const uint64_t delta = static_cast<uint64_t>(target - capacity);
                    capacity = target;
                    const uint64_t reserved =
                        rasterDependencyStats().scratch_reserved_bytes.fetch_add(
                            delta, std::memory_order_relaxed)
                        + delta;
                    updateAtomicMax(
                        rasterDependencyStats().scratch_peak_reserved_bytes, reserved);
                }
                return reinterpret_cast<char*>(buffer.data_ptr());
            };
        }
        for (size_t i = 0; i < LogicalCapacityCount; ++i) {
            lease.logical_capacities[i] = selected->logical_capacities[i];
            lease.reserve_logical[i] = [selected, i](const uint32_t required) {
                auto& capacity = selected->logical_capacities[i];
                if (required > capacity) {

                    const uint64_t divisor = i == 0 ? 4u : 8u;
                    const uint64_t required64 = required;
                    const uint64_t required_headroom =
                        std::max<uint64_t>(1024u, required64 / divisor);
                    const uint64_t current_headroom =
                        std::max<uint64_t>(1024u,
                                           static_cast<uint64_t>(capacity) / divisor);
                    const uint64_t wanted = required64 + required_headroom;
                    const uint64_t grown = static_cast<uint64_t>(capacity)
                        + current_headroom;
                    uint64_t target = std::max(wanted, grown);
                    constexpr uint64_t alignment = 256;
                    target = ((target + alignment - 1) / alignment) * alignment;
                    if (target > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                        throw std::overflow_error("raster logical capacity overflow");
                    }
                    capacity = static_cast<uint32_t>(target);
                    rasterDependencyStats().raster_capacity_grows.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return capacity;
            };
        }
        lease.summary_device = reinterpret_cast<RasterCountSummary*>(
            selected->summary_device.data_ptr());
        lease.read_summary = [selected]() -> const RasterCountSummary& {
            const auto stream = c10::cuda::getCurrentCUDAStream(selected->device_index);
            const cudaError_t copy_status = cudaMemcpyAsync(
                selected->summary_host.data_ptr(),
                selected->summary_device.data_ptr(),
                sizeof(RasterCountSummary),
                cudaMemcpyDeviceToHost,
                stream.stream());
            if (copy_status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(copy_status));
            }
            selected->summary_ready_event.record(stream);
            selected->summary_ready_event.synchronize();
            rasterDependencyStats().packed_summary_sync_count.fetch_add(
                1, std::memory_order_relaxed);
            return *reinterpret_cast<const RasterCountSummary*>(
                selected->summary_host.data_ptr());
        };
        return lease;
    }

    void finishForward(const int64_t slot_id, const bool retain_for_backward)
    {
        finish(slot_id, !retain_for_backward);
    }

    void finishBackward(const int64_t slot_id)
    {
        finish(slot_id, true);
    }

    private:
    static bool exclusivelyOwnedByPool(const Slot& slot)
    {
        for (const auto& buffer : slot.buffers) {
            if (buffer.defined() && buffer.use_count() != 1) {
                return false;
            }
        }
        return true;
    }

    void finish(const int64_t slot_id, const bool make_available)
    {
        if (slot_id == kInvalidSlot) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) {
            return;
        }

        auto& slot = slots_[static_cast<size_t>(slot_id)];
        const auto stream = c10::cuda::getCurrentCUDAStream(slot.device_index);
        slot.ready_event.record(stream);
        if (make_available) {
            slot.in_use = false;
        }
    }

    std::mutex mutex_;
    std::deque<Slot> slots_;
};

} 
