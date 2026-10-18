/*
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#include "gs/rgb_ingest.cuh"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "gs/nvtx_utils.hpp"

namespace gs {
namespace {

constexpr size_t kStagingSlotCount = 3;

struct StagingSlot {
    uint8_t* device_ptr = nullptr;
    size_t capacity = 0;
    at::cuda::CUDAEvent ready;
};

__global__ void ingestRgbKernel(const uint8_t* input,
                                const size_t row_stride,
                                const int width,
                                const int height,
                                const bool swap_rb,
                                float* output)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    const int y = pixel / width;
    const int x = pixel - y * width;
    const uint8_t* rgb = input + static_cast<size_t>(y) * row_stride
                         + static_cast<size_t>(x) * 3u;
    const float scale = 1.0f / 255.0f;
    const float c0 = static_cast<float>(rgb[swap_rb ? 2 : 0]) * scale;
    const float c1 = static_cast<float>(rgb[1]) * scale;
    const float c2 = static_cast<float>(rgb[swap_rb ? 0 : 2]) * scale;
    output[pixel] = c0;
    output[pixel_count + pixel] = c1;
    output[2 * pixel_count + pixel] = c2;
}

void launchIngestRgb(const uint8_t* input,
                     const size_t row_stride,
                     const int width,
                     const int height,
                     const RgbChannelOrder order,
                     float* output,
                     cudaStream_t stream)
{
    constexpr int threads = 256;
    const int64_t pixel_count = static_cast<int64_t>(width) * height;
    const int blocks = static_cast<int>((pixel_count + threads - 1) / threads);
    ingestRgbKernel<<<blocks, threads, 0, stream>>>(input,
                                                    row_stride,
                                                    width,
                                                    height,
                                                    order == RgbChannelOrder::BGR,
                                                    output);
}

class RgbIngestContext {
    public:
    explicit RgbIngestContext(const int device_index) :
            device_index_(device_index),
            stream_(c10::cuda::getStreamFromPool(false, device_index))
    {
    }

    ~RgbIngestContext()
    {
        try {
            stream_.synchronize();
            c10::cuda::CUDAGuard guard(device_index_);
            for (auto& slot : staging_) {
                if (slot.device_ptr) {
                    cudaFree(slot.device_ptr);
                    slot.device_ptr = nullptr;
                    slot.capacity = 0;
                }
            }
        }
        catch (...) {
        }
    }

    RgbIngestResult run(const RgbInputView& input, const RgbIngestBackend backend)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        c10::cuda::CUDAGuard device_guard(device_index_);
        const auto consumer_stream = c10::cuda::getCurrentCUDAStream(device_index_);
        c10::cuda::CUDAStreamGuard ingest_guard(stream_);

        torch::Tensor output = torch::empty(
            {3, input.height, input.width},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::Device(torch::kCUDA, device_index_)));

        const uint8_t* kernel_input = nullptr;
        size_t kernel_stride = input.row_stride;
        StagingSlot* staging_slot = nullptr;
        const size_t row_bytes = static_cast<size_t>(input.width) * 3u;
        const size_t packed_bytes = row_bytes * static_cast<size_t>(input.height);

        if (backend == RgbIngestBackend::Mapped) {
            kernel_input = input.mapped_device_ptr;
        }
        else {
            staging_slot = &staging_[next_staging_];
            next_staging_ = (next_staging_ + 1u) % staging_.size();
            staging_slot->ready.block(stream_);
            if (staging_slot->capacity < packed_bytes) {
                staging_slot->ready.synchronize();
                if (staging_slot->device_ptr) {
                    cudaFree(staging_slot->device_ptr);
                }
                const cudaError_t allocation_status =
                    cudaMalloc(reinterpret_cast<void**>(&staging_slot->device_ptr), packed_bytes);
                if (allocation_status != cudaSuccess) {
                    staging_slot->device_ptr = nullptr;
                    staging_slot->capacity = 0;
                    throw std::runtime_error(
                        std::string("GS RGB staging allocation failed: ")
                        + cudaGetErrorString(allocation_status));
                }
                staging_slot->capacity = packed_bytes;
            }

            const cudaError_t copy_status = cudaMemcpy2DAsync(staging_slot->device_ptr,
                                                               row_bytes,
                                                               input.host_ptr,
                                                               input.row_stride,
                                                               row_bytes,
                                                               input.height,
                                                               cudaMemcpyHostToDevice,
                                                               stream_.stream());
            if (copy_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("GS RGB staging copy failed: ")
                    + cudaGetErrorString(copy_status));
            }

            if (input.record_cuda_use) {
                input.record_cuda_use(reinterpret_cast<void*>(stream_.stream()));
            }
            kernel_input = staging_slot->device_ptr;
            kernel_stride = row_bytes;
        }

        launchIngestRgb(kernel_input,
                        kernel_stride,
                        input.width,
                        input.height,
                        input.channel_order,
                        output.data_ptr<float>(),
                        stream_.stream());
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("GS RGB ingest kernel launch failed: ")
                + cudaGetErrorString(launch_status));
        }

        if (backend == RgbIngestBackend::Mapped && input.record_cuda_use) {

            input.record_cuda_use(reinterpret_cast<void*>(stream_.stream()));
        }
        if (staging_slot) {
            staging_slot->ready.record(stream_);
        }

        at::cuda::CUDAEvent ingest_done;
        ingest_done.record(stream_);
        ingest_done.block(consumer_stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), consumer_stream);

        return {std::move(output), backend, {}};
    }

    private:
    int device_index_ = -1;
    c10::cuda::CUDAStream stream_;
    std::array<StagingSlot, kStagingSlotCount> staging_;
    size_t next_staging_ = 0;
    std::mutex mutex_;
};

RgbIngestContext& contextForDevice(const int device_index)
{
    static std::mutex contexts_mutex;
    static std::map<int, std::unique_ptr<RgbIngestContext>> contexts;
    std::lock_guard<std::mutex> lock(contexts_mutex);
    auto& context = contexts[device_index];
    if (!context) {
        context = std::make_unique<RgbIngestContext>(device_index);
    }
    return *context;
}

RgbIngestBackend selectBackend(const RgbInputView& input, const RgbIngestMode mode)
{
    if (mode == RgbIngestMode::HostCopy) {
        return RgbIngestBackend::HostCopy;
    }
    if (mode == RgbIngestMode::Mapped) {
        if (input.memory_kind != RgbInputMemoryKind::RegisteredMapped
            || !input.mapped_device_ptr || !input.record_cuda_use) {
            throw std::runtime_error(
                "GOFUSION_GS_RGB_INGEST=mapped requires a mapped RGB slot and fence callback");
        }
        return RgbIngestBackend::Mapped;
    }
    if (mode == RgbIngestMode::Staged) {
        if ((input.memory_kind != RgbInputMemoryKind::PinnedStaging
             && input.memory_kind != RgbInputMemoryKind::RegisteredMapped)
            || !input.record_cuda_use) {
            throw std::runtime_error(
                "GOFUSION_GS_RGB_INGEST=staged requires pinned/registered RGB input and a fence callback");
        }
        return RgbIngestBackend::Staged;
    }

    if (input.memory_kind == RgbInputMemoryKind::RegisteredMapped
        && input.mapped_device_ptr && input.record_cuda_use) {
        return RgbIngestBackend::Mapped;
    }
    if (input.memory_kind == RgbInputMemoryKind::PinnedStaging
        && input.record_cuda_use) {
        return RgbIngestBackend::Staged;
    }
    return RgbIngestBackend::HostCopy;
}

torch::Tensor hostRgbFloatChw(const RgbInputView& input)
{
    auto image = torch::from_blob(const_cast<uint8_t*>(input.host_ptr),
                                  {input.height, input.width, 3},
                                  {static_cast<int64_t>(input.row_stride), 3, 1},
                                  torch::kUInt8);
    if (input.channel_order == RgbChannelOrder::BGR) {
        const auto channels = torch::tensor(
            {2, 1, 0}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        image = image.index_select(2, channels);
    }
    auto output = image.permute({2, 0, 1})
                      .to(torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(torch::kCPU)
                              .pinned_memory(true));
    output.mul_(1.0f / 255.0f);
    output.clamp_(0.0f, 1.0f);
    return output;
}

RgbIngestResult hostCopyIngest(const RgbInputView& input)
{
    GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/rgb_ingest_host_copy");
    auto cpu = hostRgbFloatChw(input);
    auto output = cpu.to(torch::kCUDA, true);
    return {std::move(output), RgbIngestBackend::HostCopy, std::move(cpu)};
}

} 

RgbIngestMode configuredRgbIngestMode()
{
    static const RgbIngestMode mode = []() {
        const char* value = std::getenv("GOFUSION_GS_RGB_INGEST");
        if (!value || std::strcmp(value, "auto") == 0) {
            return RgbIngestMode::Auto;
        }
        if (std::strcmp(value, "mapped") == 0) {
            return RgbIngestMode::Mapped;
        }
        if (std::strcmp(value, "staged") == 0) {
            return RgbIngestMode::Staged;
        }
        if (std::strcmp(value, "host_copy") == 0) {
            return RgbIngestMode::HostCopy;
        }
        throw std::runtime_error(
            std::string("Invalid GOFUSION_GS_RGB_INGEST value: ") + value);
    }();
    return mode;
}

RgbIngestResult ingestRgb(const RgbInputView& input)
{
    return ingestRgb(input, configuredRgbIngestMode());
}

RgbIngestResult ingestRgb(const RgbInputView& input, const RgbIngestMode mode)
{
    if (!input.valid()) {
        throw std::invalid_argument("GS RGB ingest received an invalid RGB8 input view");
    }
    if (input.width > std::numeric_limits<int>::max() / input.height
        || input.row_stride
               > std::numeric_limits<size_t>::max()
                     / static_cast<size_t>(input.height)) {
        throw std::overflow_error("GS RGB ingest dimensions overflow address arithmetic");
    }
    const auto backend = selectBackend(input, mode);
    RgbIngestResult result;
    if (backend == RgbIngestBackend::HostCopy) {
        result = hostCopyIngest(input);
    }
    else {
        GOFUSION_NVTX_RANGE(backend == RgbIngestBackend::Mapped
                                ? "GOFusion/GSUpdater/rgb_ingest_mapped"
                                : "GOFusion/GSUpdater/rgb_ingest_staged");
        int device_index = -1;
        const cudaError_t device_status = cudaGetDevice(&device_index);
        if (device_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("GS RGB ingest cannot resolve CUDA device: ")
                + cudaGetErrorString(device_status));
        }
        result = contextForDevice(device_index).run(input, backend);
    }

    return result;
}

torch::Tensor makeRgbCpuSnapshot(const RgbInputView& input)
{
    if (!input.valid()) {
        throw std::invalid_argument("Cannot snapshot an invalid RGB input view");
    }
    auto fp32 = hostRgbFloatChw(input);
    return fp32.to(torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCPU)
                       .pinned_memory(true));
}

torch::Tensor makeRgbCpuSnapshot(const torch::Tensor& cuda_rgb_chw)
{
    if (!cuda_rgb_chw.defined() || !cuda_rgb_chw.is_cuda()
        || cuda_rgb_chw.scalar_type() != torch::kFloat32
        || cuda_rgb_chw.dim() != 3 || cuda_rgb_chw.size(0) != 3) {
        throw std::invalid_argument(
            "RGB CPU snapshot requires a CUDA float32 tensor shaped [3,H,W]");
    }
    auto snapshot = torch::empty(
        cuda_rgb_chw.sizes(),
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCPU)
            .pinned_memory(true));
    snapshot.copy_(cuda_rgb_chw, false);
    return snapshot;
}

} 
