/**
 * OKVIS2-X - local per-submap Gaussian Fisher export.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#ifndef INCLUDE_OKVIS_GSFISHERSNAPSHOT_HPP
#define INCLUDE_OKVIS_GSFISHERSNAPSHOT_HPP

#include <cstdint>
#include <filesystem>
#include <string>

#include <torch/torch.h>

namespace gs {
struct PlyTensorsCPU;
}

namespace okvis {

struct GsFisherSnapshotCPU {
  uint64_t submapId = 0;
  uint64_t contentRevision = 1;
  int64_t gaussianCount = 0;
  torch::Tensor xyzDiag;
  torch::Tensor opacityDiag;
  uint64_t sourceCacheVersion = 0;
  uint64_t historyViewCount = 0;
  uint64_t estimateUpdateCount = 0;
  float historyEmaAlpha = 0.2f;
  std::string evaluatedStage = "final_online_state";
  std::string alignedStage = "same_content";
  bool recomputedAfterGlobalOptimization = false;
  std::string unavailableReason;

  bool available() const
  {
    return unavailableReason.empty() && xyzDiag.defined() && opacityDiag.defined();
  }
};

struct GsFisherExportOptions {
  float xyzWeight = 1.0f;
  float opacityWeight = 1.0f;
  float eigRegularizer = 1.0e-6f;
  float displayRegularizer = 0.1f;
};

bool writeGsFisherSnapshot(const std::filesystem::path& iterationDirectory,
                           const std::filesystem::path& gsPlyPath,
                           const gs::PlyTensorsCPU& finalGsTensors,
                           const GsFisherSnapshotCPU& snapshot,
                           const GsFisherExportOptions& options,
                           std::string* error = nullptr);

} 

#endif 
