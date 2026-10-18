/*
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#include "gs/preview_converter.cuh"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "gs/nvtx_utils.hpp"

namespace gs {
namespace {

constexpr size_t kPreviewSlotCount = 2;

struct PreviewStatsAtomic {
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> staging_grows{0};
    std::atomic<uint64_t> bytes_d2h{0};
};

PreviewStatsAtomic& previewStatsAtomic()
{
    static PreviewStatsAtomic stats;
    return stats;
}

} 

namespace detail {

struct PreviewRgb8Slot {
    torch::Tensor device;
    torch::Tensor host;
    size_t capacity = 0;
    at::cuda::CUDAEvent ready;
    std::atomic<bool> in_use{false};
    bool ready_recorded = false;
};

} 

namespace {

__global__ void previewFloatHwcToRgb8Kernel(const float* input,
                                             uint8_t* output,
                                             const int64_t values)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= values) {
        return;
    }
    const float value = fminf(1.0f, fmaxf(0.0f, input[index]));
    output[index] = static_cast<uint8_t>(value * 255.0f);
}

class PreviewConversionContext {
public:
    explicit PreviewConversionContext(const int device_index) : device_index_(device_index)
    {
        for (auto& slot : slots_) {
            slot = std::make_shared<detail::PreviewRgb8Slot>();
        }
    }

    ~PreviewConversionContext()
    {
        try {
            c10::cuda::CUDAGuard guard(device_index_);
            for (auto& slot : slots_) {
                if (slot && slot->ready_recorded) {
                    slot->ready.synchronize();
                }
            }
        }
        catch (...) {
        }
    }

    PreviewRgb8CpuFrame stage(const torch::Tensor& input)
    {
        GOFUSION_NVTX_RANGE("GOFusion/Preview/stage_rgb8_d2h");
        c10::cuda::CUDAGuard guard(device_index_);
        const auto stream = c10::cuda::getCurrentCUDAStream(device_index_);
        c10::cuda::CUDAStreamGuard stream_guard(stream);

        std::shared_ptr<detail::PreviewRgb8Slot> slot;
        const size_t first = next_slot_.fetch_add(1, std::memory_order_relaxed);
        for (size_t offset = 0; offset < slots_.size(); ++offset) {
            const size_t index = (first + offset) % slots_.size();
            bool expected = false;
            if (slots_[index]->in_use.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                slot = slots_[index];
                break;
            }
        }
        if (!slot) {
            return {};
        }

        auto lease = std::shared_ptr<detail::PreviewRgb8Slot>(
            slot.get(),
            [slot](detail::PreviewRgb8Slot*) {
                slot->in_use.store(false, std::memory_order_release);
            });

        const int height = static_cast<int>(input.size(0));
        const int width = static_cast<int>(input.size(1));
        const int64_t value_count = input.numel();
        const size_t bytes = static_cast<size_t>(value_count);
        if (slot->capacity < bytes) {
            if (slot->ready_recorded) {
                slot->ready.synchronize();
            }
            ensureCapacity(*slot, bytes);
        } else if (slot->ready_recorded) {

            slot->ready.block(stream);
        }

        constexpr int threads = 256;
        const int blocks = static_cast<int>((value_count + threads - 1) / threads);
        previewFloatHwcToRgb8Kernel<<<blocks, threads, 0, stream.stream()>>>(
            input.data_ptr<float>(), slot->device.data_ptr<uint8_t>(), value_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto device_view = slot->device.narrow(0, 0, value_count);
        auto host_view = slot->host.narrow(0, 0, value_count);
        host_view.copy_(device_view, true);
        slot->ready.record(stream);
        slot->ready_recorded = true;

        auto& stats = previewStatsAtomic();
        stats.submitted.fetch_add(1, std::memory_order_relaxed);
        stats.bytes_d2h.fetch_add(bytes, std::memory_order_relaxed);
        PreviewRgb8CpuFrame frame;
        frame.slot = std::move(lease);
        frame.height = height;
        frame.width = width;
        return frame;
    }

private:
    void ensureCapacity(detail::PreviewRgb8Slot& slot, const size_t required)
    {
        if (slot.capacity >= required) {
            return;
        }

        const size_t growth = std::max<size_t>(4096, slot.capacity / 4);
        size_t target = std::max(required, slot.capacity + growth);
        constexpr size_t alignment = 256;
        target = ((target + alignment - 1) / alignment) * alignment;
        slot.device = torch::empty(
            {static_cast<int64_t>(target)},
            torch::TensorOptions().dtype(torch::kUInt8).device(
                torch::Device(torch::kCUDA, device_index_)));
        slot.host = torch::empty(
            {static_cast<int64_t>(target)},
            torch::TensorOptions()
                .dtype(torch::kUInt8)
                .device(torch::kCPU)
                .pinned_memory(true));
        slot.capacity = target;
        previewStatsAtomic().staging_grows.fetch_add(1, std::memory_order_relaxed);
    }

    int device_index_ = -1;
    std::array<std::shared_ptr<detail::PreviewRgb8Slot>, kPreviewSlotCount> slots_;
    std::atomic<size_t> next_slot_{0};
};

PreviewConversionContext& contextForDevice(const int device_index)
{
    static std::mutex contexts_mutex;
    static std::map<int, std::unique_ptr<PreviewConversionContext>> contexts;
    std::lock_guard<std::mutex> lock(contexts_mutex);
    auto& context = contexts[device_index];
    if (!context) {
        context = std::make_unique<PreviewConversionContext>(device_index);
    }
    return *context;
}

} 

PreviewRgb8CpuFrame stagePreviewRgbHwcFloatToCpu(
    const torch::Tensor& rendered_rgb_hwc)
{
    if (!rendered_rgb_hwc.defined() || !rendered_rgb_hwc.is_cuda()
        || rendered_rgb_hwc.scalar_type() != torch::kFloat32
        || rendered_rgb_hwc.dim() != 3 || rendered_rgb_hwc.size(2) != 3
        || rendered_rgb_hwc.size(0) <= 0 || rendered_rgb_hwc.size(1) <= 0
        || !rendered_rgb_hwc.is_contiguous()) {
        throw std::invalid_argument(
            "Preview conversion requires contiguous CUDA float32 HWC RGB input");
    }
    return contextForDevice(rendered_rgb_hwc.get_device())
        .stage(rendered_rgb_hwc);
}

cv::Mat consumePreviewRgbCpu(PreviewRgb8CpuFrame frame)
{
    if (!frame.valid()) {
        return {};
    }
    GOFUSION_NVTX_RANGE("GOFusion/Preview/wait_cpu_publish");
    auto slot = std::move(frame.slot);
    slot->ready.synchronize();
    previewStatsAtomic().completed.fetch_add(1, std::memory_order_relaxed);
    return cv::Mat(frame.height,
                   frame.width,
                   CV_8UC3,
                   slot->host.data_ptr<uint8_t>())
        .clone();
}

PreviewConversionStatsSnapshot previewConversionStats()
{
    const auto& stats = previewStatsAtomic();
    PreviewConversionStatsSnapshot out;
    out.submitted = stats.submitted.load(std::memory_order_relaxed);
    out.completed = stats.completed.load(std::memory_order_relaxed);
    out.staging_grows = stats.staging_grows.load(std::memory_order_relaxed);
    out.bytes_d2h = stats.bytes_d2h.load(std::memory_order_relaxed);
    return out;
}

} 
