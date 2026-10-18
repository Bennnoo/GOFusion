/**
 * OKVIS2-X - reusable cross-platform frame staging storage.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENSE file for details.
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#ifndef INCLUDE_OKVIS_SHARED_FRAME_SLOT_HPP_
#define INCLUDE_OKVIS_SHARED_FRAME_SLOT_HPP_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <okvis/Measurements.hpp>

namespace okvis {

using SharedFrameMemoryKind = FrameBufferMemoryKind;

const char* sharedFrameMemoryKindName(SharedFrameMemoryKind kind) noexcept;

class SharedFrameSlot final : public FrameBufferLifetime {
public:
  SharedFrameSlot();
  ~SharedFrameSlot() override;

  SharedFrameSlot(const SharedFrameSlot&) = delete;
  SharedFrameSlot& operator=(const SharedFrameSlot&) = delete;

  bool prepare(int rows, int cols, int type);
  bool reusable() const noexcept;

  cv::Mat& mat() noexcept { return storage_; }
  const cv::Mat& mat() const noexcept { return storage_; }
  SharedFrameMemoryKind memoryKind() const noexcept override { return memoryKind_; }
  const void* hostData() const noexcept override { return storage_.data; }
  const void* mappedDeviceData() const noexcept override {
    return mappedDeviceAddress_;
  }
  size_t capacityBytes() const noexcept override { return capacityBytes_; }

  void recordCudaUse(void* native_stream) noexcept override;

private:
  void releaseCudaRegistration() noexcept;
  bool cudaFenceComplete() const noexcept;

  cv::Mat storage_;
  size_t capacityBytes_ = 0;
  SharedFrameMemoryKind memoryKind_ = SharedFrameMemoryKind::OrdinaryHost;
  void* registeredAddress_ = nullptr;
  void* mappedDeviceAddress_ = nullptr;
  size_t registeredBytes_ = 0;
  void* cudaEvent_ = nullptr;
};

class SharedFrameSlotRing {
public:
  explicit SharedFrameSlotRing(size_t slot_count = 6);

  std::shared_ptr<SharedFrameSlot> acquire(int rows, int cols, int type);
  uint64_t droppedCount() const noexcept {
    return droppedCount_.load(std::memory_order_relaxed);
  }
  size_t highWatermark() const noexcept {
    return highWatermark_.load(std::memory_order_relaxed);
  }
  size_t size() const noexcept { return slots_.size(); }

private:
  std::vector<std::shared_ptr<SharedFrameSlot>> slots_;
  size_t nextSlot_ = 0;
  std::mutex mutex_;
  std::atomic<uint64_t> droppedCount_{0};
  std::atomic<size_t> highWatermark_{0};
};

}  

#endif  
