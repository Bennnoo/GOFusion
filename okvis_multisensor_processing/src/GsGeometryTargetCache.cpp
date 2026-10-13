/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "okvis/GsGeometryTargetCache.hpp"

#include "gs/geometry_supervision.cuh"
#include <zstd.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace okvis {
namespace {

void packMask(const bool* source, const size_t count, uint8_t* destination)
{
  for (size_t i = 0; i < count; i += 8) {
    uint8_t bits = 0;
    for (size_t bit = 0; bit < 8 && i + bit < count; ++bit) {
      bits |= static_cast<uint8_t>(source[i + bit]) << bit;
    }
    destination[i / 8] = bits;
  }
}

void unpackMask(const uint8_t* source, const size_t count, bool* destination)
{
  for (size_t i = 0; i < count; i += 8) {
    const uint8_t bits = source[i / 8];
    for (size_t bit = 0; bit < 8 && i + bit < count; ++bit) {
      destination[i + bit] = (bits >> bit) & 1u;
    }
  }
}

} 

struct GsGeometryTargetCache::Impl {
  struct Entry {

    std::unique_ptr<uint8_t[]> data;
    size_t normalBytes = 0;
    int64_t height = 0;
    int64_t width = 0;
    bool compressed = false;
  };

  explicit Impl(const size_t viewCount)
      : entries(viewCount),
        encoder(ZSTD_createCCtx(), ZSTD_freeCCtx),
        decoder(ZSTD_createDCtx(), ZSTD_freeDCtx)
  {
    if (!encoder || !decoder) {
      throw std::runtime_error("Cannot allocate GS geometry compression context");
    }
  }

  std::vector<Entry> entries;
  std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> encoder;
  std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> decoder;
  std::vector<uint8_t> scratch;
  size_t cachedViews = 0;
  size_t payloadBytes = 0;
};

GsGeometryTargetCache::GsGeometryTargetCache(const size_t viewCount)
    : impl_(std::make_unique<Impl>(viewCount)) {}

GsGeometryTargetCache::~GsGeometryTargetCache() = default;

size_t GsGeometryTargetCache::cachedViews() const { return impl_->cachedViews; }
size_t GsGeometryTargetCache::payloadBytes() const { return impl_->payloadBytes; }

gs::GeometryTarget GsGeometryTargetCache::get(const size_t index,
                                            const torch::Tensor& depthCpu,
                                            const gs::Camera& camera)
{
  auto& entry = impl_->entries.at(index);
  if (!entry.data) {
    auto target = gs::makeGeometryTarget(
        depthCpu, camera, 0.0f, std::numeric_limits<float>::infinity(),
        torch::Device(torch::kCPU));
    const size_t normalBytes = target.normal.nbytes();
    const size_t pixels = target.depth.numel();
    const size_t maskBytes = (pixels + 7) / 8;
    impl_->scratch.resize(ZSTD_compressBound(normalBytes));
    const size_t compressedBytes = ZSTD_compressCCtx(
        impl_->encoder.get(), impl_->scratch.data(), impl_->scratch.size(),
        target.normal.data_ptr<float>(), normalBytes, 1);
    if (ZSTD_isError(compressedBytes)) {
      throw std::runtime_error(std::string("GS geometry compression failed: ")
                               + ZSTD_getErrorName(compressedBytes));
    }

    Impl::Entry encoded;
    encoded.compressed = compressedBytes < normalBytes;
    encoded.normalBytes = encoded.compressed ? compressedBytes : normalBytes;
    encoded.height = target.depth.size(1);
    encoded.width = target.depth.size(2);
    const size_t bytes = encoded.normalBytes + 2 * maskBytes;

    encoded.data.reset(new uint8_t[bytes]);
    const void* normalSource = encoded.compressed
        ? static_cast<const void*>(impl_->scratch.data())
        : static_cast<const void*>(target.normal.data_ptr<float>());
    std::memcpy(encoded.data.get(), normalSource, encoded.normalBytes);
    packMask(target.depth_mask.data_ptr<bool>(), pixels,
             encoded.data.get() + encoded.normalBytes);
    packMask(target.normal_mask.data_ptr<bool>(), pixels,
             encoded.data.get() + encoded.normalBytes + maskBytes);
    entry = std::move(encoded);
    ++impl_->cachedViews;
    impl_->payloadBytes += bytes;

    return target;
  }

  TORCH_CHECK(depthCpu.device().is_cpu() && depthCpu.scalar_type() == torch::kFloat32
                  && depthCpu.dim() == 3 && depthCpu.size(0) == 1
                  && depthCpu.size(1) == entry.height && depthCpu.size(2) == entry.width,
              "GS geometry cache depth shape/type changed");
  gs::GeometryTarget target;
  target.depth = depthCpu.contiguous();
  target.normal = torch::empty({3, entry.height, entry.width}, target.depth.options());
  target.depth_mask = torch::empty_like(target.depth, target.depth.options().dtype(torch::kBool));
  target.normal_mask = torch::empty_like(target.depth_mask);
  const size_t normalBytes = target.normal.nbytes();
  if (entry.compressed) {
    const size_t decodedBytes = ZSTD_decompressDCtx(
        impl_->decoder.get(), target.normal.data_ptr<float>(), normalBytes,
        entry.data.get(), entry.normalBytes);
    if (ZSTD_isError(decodedBytes) || decodedBytes != normalBytes) {
      throw std::runtime_error("GS geometry decompression failed or returned an invalid size");
    }
  } else {
    std::memcpy(target.normal.data_ptr<float>(), entry.data.get(), normalBytes);
  }
  const size_t pixels = target.depth.numel();
  const size_t maskBytes = (pixels + 7) / 8;
  unpackMask(entry.data.get() + entry.normalBytes, pixels, target.depth_mask.data_ptr<bool>());
  unpackMask(entry.data.get() + entry.normalBytes + maskBytes, pixels,
             target.normal_mask.data_ptr<bool>());
  return target;
}

} 
