/**
 * OKVIS2-X - local per-submap Gaussian Fisher export.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#include <okvis/GsFisherSnapshot.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>
#include <gs/gaussian.cuh>

namespace okvis {
namespace {

bool isLittleEndian()
{
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

void replaceFile(const std::filesystem::path& temporary,
                 const std::filesystem::path& destination)
{
  std::error_code ec;
  std::filesystem::rename(temporary, destination, ec);
  if (!ec) {
    return;
  }
  std::filesystem::remove(destination, ec);
  ec.clear();
  std::filesystem::rename(temporary, destination, ec);
  if (ec) {
    throw std::runtime_error("failed to publish " + destination.string() + ": " + ec.message());
  }
}

void removeFileNoThrow(const std::filesystem::path& path)
{
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

void requireFloatMatrix(const torch::Tensor& tensor,
                        const char* name,
                        const int64_t rows,
                        const int64_t columns)
{
  if (!tensor.defined() || !tensor.device().is_cpu()
      || tensor.scalar_type() != torch::kFloat32 || !tensor.is_contiguous()
      || tensor.dim() != 2 || tensor.size(0) != rows || tensor.size(1) != columns) {
    std::ostringstream message;
    message << name << " must be contiguous CPU float32 (" << rows << "," << columns << ")";
    throw std::invalid_argument(message.str());
  }
}

void validateGsTensors(const gs::PlyTensorsCPU& tensors, const int64_t rows)
{
  requireFloatMatrix(tensors.xyz, "gs.xyz", rows, 3);
  requireFloatMatrix(tensors.f_dc, "gs.f_dc", rows, 3);
  if (!tensors.f_rest.defined() || tensors.f_rest.dim() != 2) {
    throw std::invalid_argument("gs.f_rest must be a 2D tensor");
  }
  requireFloatMatrix(tensors.f_rest, "gs.f_rest", rows, tensors.f_rest.size(1));
  requireFloatMatrix(tensors.opacity, "gs.opacity", rows, 1);
  requireFloatMatrix(tensors.scale, "gs.scale", rows, 3);
  requireFloatMatrix(tensors.rotation, "gs.rotation", rows, 4);
}

struct WrittenSection {
  uint64_t bytes = 0;

};

WrittenSection writeFloatSection(const torch::Tensor& tensor,
                                 const std::filesystem::path& destination)
{
  const std::filesystem::path temporary = destination.string() + ".part";
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open " + temporary.string());
  }

  const float* data = tensor.data_ptr<float>();
  const size_t values = static_cast<size_t>(tensor.numel());
  stream.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(values * sizeof(float)));
  if (!stream) throw std::runtime_error("failed to write " + temporary.string());

  stream.close();
  if (!stream) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed to close " + temporary.string());
  }
  replaceFile(temporary, destination);
  return WrittenSection{static_cast<uint64_t>(values * sizeof(float))};
}

void writeJsonAtomic(const std::filesystem::path& destination,
                     const nlohmann::json& payload)
{
  const std::filesystem::path temporary = destination.string() + ".part";
  std::ofstream stream(temporary, std::ios::out | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open " + temporary.string());
  }
  stream << payload.dump(2) << '\n';
  stream.close();
  if (!stream) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed to write " + temporary.string());
  }
  replaceFile(temporary, destination);
}

nlohmann::json baseManifest(const GsFisherSnapshotCPU& snapshot)
{
  return {
      {"schema", "okfusion.gs_fisher_diag"},
      {"schema_version", 1},
      {"submap_id", snapshot.submapId},
      {"content_revision", snapshot.contentRevision},
      {"gaussian_count", snapshot.gaussianCount},
      {"gaussian_coordinate_frame", "K"},
  };
}

} 

bool writeGsFisherSnapshot(const std::filesystem::path& iterationDirectory,
                           const std::filesystem::path& gsPlyPath,
                           const gs::PlyTensorsCPU& finalGsTensors,
                           const GsFisherSnapshotCPU& snapshot,
                           const GsFisherExportOptions& options,
                           std::string* error)
{
  const std::filesystem::path fisher_directory = iterationDirectory / "fisher";
  const std::filesystem::path manifest_path = fisher_directory / "manifest.json";
  const std::filesystem::path xyz_path = fisher_directory / "xyz_diag.f32";
  const std::filesystem::path opacity_path = fisher_directory / "opacity_diag.f32";
  try {
    if (!isLittleEndian()) {
      throw std::runtime_error("raw Fisher export currently requires a little-endian host");
    }
    if (!std::isfinite(options.xyzWeight) || options.xyzWeight < 0.0f
        || !std::isfinite(options.opacityWeight) || options.opacityWeight < 0.0f
        || !std::isfinite(options.eigRegularizer) || options.eigRegularizer <= 0.0f
        || !std::isfinite(options.displayRegularizer)
        || options.displayRegularizer <= 0.0f) {
      throw std::invalid_argument("invalid Fisher display metadata");
    }
    std::filesystem::create_directories(fisher_directory);
    std::filesystem::remove(manifest_path);

    nlohmann::json manifest = baseManifest(snapshot);
    if (!snapshot.available()) {
      std::filesystem::remove(xyz_path);
      std::filesystem::remove(opacity_path);
      manifest["status"] = "unavailable";
      manifest["reason"] = snapshot.unavailableReason.empty()
                                 ? "fisher_cache_unavailable"
                                 : snapshot.unavailableReason;
      writeJsonAtomic(manifest_path, manifest);
      return true;
    }

    if (snapshot.gaussianCount <= 0) {
      throw std::invalid_argument("available Fisher snapshot has no Gaussians");
    }
    requireFloatMatrix(snapshot.xyzDiag, "fisher.xyz_diag", snapshot.gaussianCount, 3);
    requireFloatMatrix(snapshot.opacityDiag,
                       "fisher.opacity_diag",
                       snapshot.gaussianCount,
                       1);
    validateGsTensors(finalGsTensors, snapshot.gaussianCount);
    if (!std::filesystem::is_regular_file(gsPlyPath)) {
      throw std::runtime_error("bound GS PLY does not exist: " + gsPlyPath.string());
    }

    const WrittenSection xyz = writeFloatSection(snapshot.xyzDiag, xyz_path);
    const WrittenSection opacity = writeFloatSection(snapshot.opacityDiag, opacity_path);

    manifest["status"] = "available";
    manifest["estimator"] = {
        {"name", "online_training_empirical_diag_v1"},
        {"parameter_groups", {"camera_space_mean", "activated_opacity"}},
        {"history_view_count", snapshot.historyViewCount},
        {"estimate_update_count", snapshot.estimateUpdateCount},
        {"source_cache_version", snapshot.sourceCacheVersion},
        {"history_ema_alpha", snapshot.historyEmaAlpha},
        {"evaluated_stage", snapshot.evaluatedStage},
        {"aligned_stage", snapshot.alignedStage},
        {"recomputed_after_global_optimization",
         snapshot.recomputedAfterGlobalOptimization},
    };
    manifest["display_defaults"] = {
        {"xyz_reduce", "mean"},
        {"xyz_weight", options.xyzWeight},
        {"opacity_weight", options.opacityWeight},
        {"fisher_eig_regularizer", options.eigRegularizer},
        {"fisher_display_regularizer", options.displayRegularizer},
    };
    manifest["gs_binding"] = {
        {"relative_path", "../point_cloud.ply"},
        {"gaussian_count", snapshot.gaussianCount},
    };
    manifest["sections"] = {
        {"fisher_xyz_diag",
         {{"path", "xyz_diag.f32"},
          {"dtype", "float32"},
          {"endianness", "little"},
          {"layout", "C"},
          {"shape", {snapshot.gaussianCount, 3}},
          {"bytes", xyz.bytes}}},
        {"fisher_opacity_diag",
         {{"path", "opacity_diag.f32"},
          {"dtype", "float32"},
          {"endianness", "little"},
          {"layout", "C"},
          {"shape", {snapshot.gaussianCount, 1}},
          {"bytes", opacity.bytes}}},
    };
    writeJsonAtomic(manifest_path, manifest);
    return true;
  } catch (const std::exception& exception) {
    removeFileNoThrow(manifest_path.string() + ".part");
    removeFileNoThrow(xyz_path.string() + ".part");
    removeFileNoThrow(opacity_path.string() + ".part");
    removeFileNoThrow(xyz_path);
    removeFileNoThrow(opacity_path);
    try {
      std::filesystem::create_directories(fisher_directory);
      nlohmann::json manifest = baseManifest(snapshot);
      manifest["status"] = "unavailable";
      manifest["reason"] = std::string("fisher_export_failed: ") + exception.what();
      writeJsonAtomic(manifest_path, manifest);
    } catch (...) {

    }
    if (error) {
      *error = exception.what();
    }
    return false;
  }
}

} 

