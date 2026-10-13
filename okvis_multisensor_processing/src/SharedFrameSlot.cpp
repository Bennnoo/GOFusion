/**
 * OKVIS2-X - reusable cross-platform frame staging storage.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENSE file for details.
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 */

#include <okvis/SharedFrameSlot.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <glog/logging.h>

#include <cuda_runtime_api.h>

namespace okvis {
namespace {

SharedFrameMemoryKind requestedMemoryKind() noexcept {
  const char* override_value = std::getenv("GOFUSION_SHARED_FRAME_MEMORY");
  if (override_value) {
    if (std::strcmp(override_value, "ordinary") == 0) {
      return SharedFrameMemoryKind::OrdinaryHost;
    }
    if (std::strcmp(override_value, "registered") == 0
        || std::strcmp(override_value, "coherent") == 0) {
      return SharedFrameMemoryKind::CudaRegisteredMapped;
    }
    if (std::strcmp(override_value, "pinned") == 0) {
      return SharedFrameMemoryKind::CudaPinnedStaging;
    }
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) == cudaSuccess
      && cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    return properties.integrated
               ? SharedFrameMemoryKind::CudaRegisteredMapped
               : SharedFrameMemoryKind::CudaPinnedStaging;
  }
  cudaGetLastError();
  return SharedFrameMemoryKind::OrdinaryHost;
}

}  

const char* sharedFrameMemoryKindName(
    const SharedFrameMemoryKind kind) noexcept {
  switch (kind) {
    case SharedFrameMemoryKind::CudaRegisteredMapped:
      return "cuda-registered-coherent";
    case SharedFrameMemoryKind::CudaPinnedStaging:
      return "cuda-pinned-staging";
    case SharedFrameMemoryKind::OrdinaryHost:
    default:
      return "ordinary-host";
  }
}

SharedFrameSlot::SharedFrameSlot() = default;

SharedFrameSlot::~SharedFrameSlot() {
  releaseCudaRegistration();
}

bool SharedFrameSlot::cudaFenceComplete() const noexcept {
  if (!cudaEvent_) {
    return true;
  }
  const cudaError_t status =
      cudaEventQuery(reinterpret_cast<cudaEvent_t>(cudaEvent_));
  if (status == cudaSuccess) {
    return true;
  }
  if (status == cudaErrorNotReady) {
    return false;
  }
  cudaGetLastError();
  return true;
}

bool SharedFrameSlot::reusable() const noexcept {

  const bool mat_is_exclusive = !storage_.u || storage_.u->refcount <= 1;
  return mat_is_exclusive && cudaFenceComplete();
}

void SharedFrameSlot::releaseCudaRegistration() noexcept {
  if (cudaEvent_) {
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(cudaEvent_));
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cudaEvent_));
    cudaEvent_ = nullptr;
  }
  if (registeredAddress_) {
    cudaHostUnregister(registeredAddress_);
    registeredAddress_ = nullptr;
    mappedDeviceAddress_ = nullptr;
    registeredBytes_ = 0;
  }
  memoryKind_ = SharedFrameMemoryKind::OrdinaryHost;
}

bool SharedFrameSlot::prepare(const int rows,
                              const int cols,
                              const int type) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  if (!reusable()) {
    return false;
  }

  const bool shape_matches = storage_.rows == rows && storage_.cols == cols
                             && storage_.type() == type
                             && storage_.isContinuous();
  if (shape_matches) {
    return true;
  }

  releaseCudaRegistration();
  storage_.create(rows, cols, type);
  if (!storage_.isContinuous()) {
    storage_.release();
    return false;
  }
  capacityBytes_ = storage_.total() * storage_.elemSize();
  memoryKind_ = requestedMemoryKind();

  if (memoryKind_ != SharedFrameMemoryKind::OrdinaryHost) {
    unsigned int flags = cudaHostRegisterPortable;
    if (memoryKind_ == SharedFrameMemoryKind::CudaRegisteredMapped) {
      int device = 0;
      cudaDeviceProp properties{};
      if (cudaGetDevice(&device) != cudaSuccess
          || cudaGetDeviceProperties(&properties, device) != cudaSuccess
          || !properties.canMapHostMemory) {
        cudaGetLastError();
        memoryKind_ = SharedFrameMemoryKind::CudaPinnedStaging;
      }
    }
    if (memoryKind_ == SharedFrameMemoryKind::CudaRegisteredMapped) {
      flags |= cudaHostRegisterMapped;
    }
    const cudaError_t register_status =
        cudaHostRegister(storage_.data, capacityBytes_, flags);
    if (register_status == cudaSuccess) {
      registeredAddress_ = storage_.data;
      registeredBytes_ = capacityBytes_;
      if (memoryKind_ == SharedFrameMemoryKind::CudaRegisteredMapped) {
        void* device_address = nullptr;
        const cudaError_t mapped_status =
            cudaHostGetDevicePointer(&device_address, storage_.data, 0);
        if (mapped_status == cudaSuccess) {
          mappedDeviceAddress_ = device_address;
        } else {
          cudaGetLastError();
          cudaHostUnregister(registeredAddress_);
          registeredAddress_ = nullptr;
          registeredBytes_ = 0;
          memoryKind_ = SharedFrameMemoryKind::CudaPinnedStaging;
          const cudaError_t pinned_status = cudaHostRegister(
              storage_.data, capacityBytes_, cudaHostRegisterPortable);
          if (pinned_status == cudaSuccess) {
            registeredAddress_ = storage_.data;
            registeredBytes_ = capacityBytes_;
          } else {
            cudaGetLastError();
            memoryKind_ = SharedFrameMemoryKind::OrdinaryHost;
          }
        }
      }
      cudaEvent_t event = nullptr;
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming)
          == cudaSuccess) {
        cudaEvent_ = reinterpret_cast<void*>(event);
      } else {
        cudaGetLastError();

        cudaHostUnregister(registeredAddress_);
        registeredAddress_ = nullptr;
        mappedDeviceAddress_ = nullptr;
        registeredBytes_ = 0;
        memoryKind_ = SharedFrameMemoryKind::OrdinaryHost;
      }
    } else {
      cudaGetLastError();
      memoryKind_ = SharedFrameMemoryKind::OrdinaryHost;
      static std::atomic_bool warned{false};
      if (!warned.exchange(true)) {
        LOG(WARNING) << "SharedFrameSlot: CUDA host registration failed; "
                        "falling back to ordinary host memory";
      }
    }
  }

  static std::atomic_bool logged_backend{false};
  if (!logged_backend.exchange(true)) {
    LOG(INFO) << "SharedFrameSlot backend="
              << sharedFrameMemoryKindName(memoryKind_)
              << " first_slot_bytes=" << capacityBytes_;
  }
  return true;
}

void SharedFrameSlot::recordCudaUse(void* native_stream) noexcept {
  if (!cudaEvent_) {
    return;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(native_stream);
  if (cudaEventRecord(reinterpret_cast<cudaEvent_t>(cudaEvent_), stream)
      != cudaSuccess) {
    cudaGetLastError();
  }
}

SharedFrameSlotRing::SharedFrameSlotRing(const size_t slot_count) {
  slots_.reserve(std::max<size_t>(1, slot_count));
  for (size_t i = 0; i < std::max<size_t>(1, slot_count); ++i) {
    slots_.push_back(std::make_shared<SharedFrameSlot>());
  }
}

std::shared_ptr<SharedFrameSlot> SharedFrameSlotRing::acquire(
    const int rows,
    const int cols,
    const int type) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t in_use = 0;
  for (const auto& slot : slots_) {
    if (slot.use_count() != 1 || !slot->reusable()) {
      ++in_use;
    }
  }
  for (size_t offset = 0; offset < slots_.size(); ++offset) {
    const size_t index = (nextSlot_ + offset) % slots_.size();
    auto& slot = slots_[index];
    if (slot.use_count() != 1 || !slot->reusable()) {
      continue;
    }
    if (!slot->prepare(rows, cols, type)) {
      continue;
    }
    nextSlot_ = (index + 1) % slots_.size();
    const size_t watermark = in_use + 1;
    size_t previous = highWatermark_.load(std::memory_order_relaxed);
    while (previous < watermark
           && !highWatermark_.compare_exchange_weak(
               previous, watermark, std::memory_order_relaxed)) {
    }
    return slot;
  }
  droppedCount_.fetch_add(1, std::memory_order_relaxed);
  return {};
}

}  
