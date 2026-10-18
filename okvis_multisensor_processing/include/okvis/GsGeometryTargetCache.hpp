/**
 * In-memory lossless CPU geometry targets for final GS optimization.
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */
#pragma once

#include <cstddef>
#include <memory>
#include <torch/types.h>

namespace gs {
struct Camera;
struct GeometryTarget;
}

namespace okvis {

class GsGeometryTargetCache {
public:
  explicit GsGeometryTargetCache(size_t viewCount);
  ~GsGeometryTargetCache();

  gs::GeometryTarget get(size_t index,
                        const torch::Tensor& depthCpu,
                        const gs::Camera& camera);

  size_t cachedViews() const;
  size_t payloadBytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} 
