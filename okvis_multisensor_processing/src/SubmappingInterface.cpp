/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#include <okvis/SubmappingInterface.hpp>
#include <okvis/DepthUtils.hpp>
#include <okvis/GsFisherSnapshot.hpp>
#include <okvis/SharedFrameSlot.hpp>
#include <se/common/eigen_utils.hpp>
#include <se/map/data.hpp>
#include <se/map/data_field.hpp>
#include <se/map/octree/allocator.hpp>
#include <se/map/octree/fetcher.hpp>
#include <se/map/octree/propagator.hpp>
#include <array>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include "se/common/point_cloud_io.hpp"
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "okvis/GsGeometryTargetCache.hpp"
#include "gs/gaussian.cuh"
#include "gs/gaussian_utils.cuh"
#include "gs/monogs_render_utils.cuh"
#include "gs/preview_converter.cuh"
#include "gs/render_utils.cuh"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

namespace {
constexpr float kGsFisherHistoryEmaAlpha = 0.2f;

class RgbFrameSourceLease {
public:
  explicit RgbFrameSourceLease(
      std::shared_ptr<okvis::FrameBufferLifetime> source)
      : source_(std::move(source)) {}

  void recordCudaUse(void* native_stream) noexcept {
    if (source_) {
      source_->recordCudaUse(native_stream);
    }
  }

  void release() noexcept { source_.reset(); }

private:
  std::shared_ptr<okvis::FrameBufferLifetime> source_;
};

bool isFatalGsIntegrationFailure(const std::exception& error)
{
  std::string message = error.what();
  std::transform(message.begin(), message.end(), message.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

  constexpr std::array<const char*, 7> recoverable_tokens = {
      "raster capacity", "rasterizer produced invalid", "packed validation",
      "scratch slot", "instance capacity overflow", "bucket count",
      "exact fallback"};
  for (const char* token : recoverable_tokens) {
    if (message.find(token) != std::string::npos) return false;
  }

  constexpr std::array<const char*, 10> fatal_tokens = {
      "illegal memory access", "device-side assert", "cuda context",
      "launch failure", "unspecified launch failure", "out of memory",
      "cublas_status", "cudnn_status", "optimizer state", "size mismatch"};
  for (const char* token : fatal_tokens) {
    if (message.find(token) != std::string::npos) return true;
  }
  return false;
}

int shDegreeFromFlattenedRest(const torch::Tensor& f_rest)
{
  if (!f_rest.defined() || f_rest.dim() != 2 || f_rest.size(1) < 0
      || f_rest.size(1) % 3 != 0) {
    throw std::invalid_argument("Gaussian f_rest has an invalid flattened layout");
  }
  const int64_t basis = f_rest.size(1) / 3 + 1;
  const int64_t order = static_cast<int64_t>(
      std::llround(std::sqrt(static_cast<double>(basis))));
  if (order <= 0 || order * order != basis) {
    throw std::invalid_argument("Gaussian f_rest does not encode a square SH basis");
  }
  return static_cast<int>(order - 1);
}

Eigen::Vector3i getVoxelColour(const okvis::SupereightMapType::DataType& voxel_data)
{
  const auto& colour = voxel_data.colour.colour;
  return Eigen::Vector3i(colour.r, colour.g, colour.b);
}

float clampUnit(const float value)
{
  return std::max(0.0f, std::min(1.0f, value));
}

Eigen::Vector3i fisherHeatRgb(const float value)
{
  static const auto turbo_lut = []() {
    std::array<Eigen::Vector3i, 256> lut{};
    cv::Mat gray(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
      gray.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
    }
    cv::Mat bgr;
    cv::applyColorMap(gray, bgr, cv::COLORMAP_TURBO);
    for (int i = 0; i < 256; ++i) {
      const cv::Vec3b colour_bgr = bgr.at<cv::Vec3b>(0, i);
      lut[static_cast<size_t>(i)] =
          Eigen::Vector3i(colour_bgr[2], colour_bgr[1], colour_bgr[0]);
    }
    return lut;
  }();
  const int idx = std::clamp(
      static_cast<int>(std::lround(255.0f * clampUnit(value))), 0, 255);
  return turbo_lut[static_cast<size_t>(idx)];
}

} 



namespace okvis {

  namespace {
    constexpr size_t kGsPreviewQueueSize = 2;
    constexpr uint64_t kRenderPoseToleranceNs = 5000000ULL; 

    constexpr uint64_t kGsMappingPoseEndpointToleranceNs = 40000000ULL; 
    constexpr uint64_t kGsMappingPoseMaxBracketNs = 80000000ULL; 
    using PoseLookup = std::map<
        uint64_t,
        Eigen::Isometry3f,
        std::less<uint64_t>,
        Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Isometry3f>>>;

    GsPreviewFrame toGsPreviewFrame(gs::DataPacket packet,
                                    const uint64_t num_submaps)
    {
      if (packet.rendered_rgb.empty() && packet.rendered_rgb_staging.valid()) {
        packet.rendered_rgb =
            gs::consumePreviewRgbCpu(std::move(packet.rendered_rgb_staging));
      }
      GsPreviewFrame preview;
      preview.id = packet.ID;
      preview.global_iter = packet.global_iter;
      preview.num_splats = packet.num_splats;
      preview.num_kf = packet.num_kf;
      preview.num_submaps = num_submaps;
      preview.fps = packet.fps;
      preview.backend_fps = packet.fps;
      preview.rgb = std::move(packet.rgb);
      preview.depth = std::move(packet.depth);
      preview.rendered_rgb = std::move(packet.rendered_rgb);
      return preview;
    }

    std::string trimString(const std::string& input)
    {
      size_t begin = 0;
      while (begin < input.size()
             && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
      }
      size_t end = input.size();
      while (end > begin
             && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
      }
      return input.substr(begin, end - begin);
    }

    bool loadFinalTrajectoryPosesFromProvider(
        const std::vector<ViSlamBackend::FinalTrajectoryPosition>& trajectoryPoses,
        PoseLookup& posesByTimestamp,
        PoseLookup& posesByStateId)
    {
      posesByTimestamp.clear();
      posesByStateId.clear();
      if (trajectoryPoses.empty()) {
        return false;
      }

      for (const auto& pose : trajectoryPoses) {
        Eigen::Quaternionf q(static_cast<float>(pose.qw),
                             static_cast<float>(pose.qx),
                             static_cast<float>(pose.qy),
                             static_cast<float>(pose.qz));
        q.normalize();

        Eigen::Isometry3f T_WS = Eigen::Isometry3f::Identity();
        T_WS.linear() = q.toRotationMatrix();
        T_WS.translation() = Eigen::Vector3f(static_cast<float>(pose.px),
                                             static_cast<float>(pose.py),
                                             static_cast<float>(pose.pz));

        posesByTimestamp[pose.timestamp_ns] = T_WS;
        posesByStateId[pose.state_id] = T_WS;
      }

      return !posesByTimestamp.empty();
    }

    bool loadFinalTrajectoryPoses(
        const std::string& trajectoryCsv,
        PoseLookup& posesByTimestamp)
    {
      std::ifstream in(trajectoryCsv);
      if (!in.good()) {
        LOG(ERROR) << "GS finalization: cannot open final trajectory CSV "
                   << trajectoryCsv;
        return false;
      }

      std::string line;
      std::getline(in, line); 
      while (std::getline(in, line)) {
        if (line.empty()) {
          continue;
        }
        std::stringstream ss(line);
        std::string tStr;
        std::string pxStr, pyStr, pzStr, qxStr, qyStr, qzStr, qwStr;
        if (!std::getline(ss, tStr, ',')) {
          continue;
        }
        if (!std::getline(ss, pxStr, ',')) {
          continue;
        }
        if (!std::getline(ss, pyStr, ',')) {
          continue;
        }
        if (!std::getline(ss, pzStr, ',')) {
          continue;
        }
        if (!std::getline(ss, qxStr, ',')) {
          continue;
        }
        if (!std::getline(ss, qyStr, ',')) {
          continue;
        }
        if (!std::getline(ss, qzStr, ',')) {
          continue;
        }
        if (!std::getline(ss, qwStr, ',')) {
          continue;
        }

        const std::string ts = trimString(tStr);
        if (ts.empty()) {
          continue;
        }

        uint64_t timestamp = 0;
        try {
          timestamp = static_cast<uint64_t>(std::stoull(ts));
        } catch (...) {
          continue;
        }

        const double px = std::stod(trimString(pxStr));
        const double py = std::stod(trimString(pyStr));
        const double pz = std::stod(trimString(pzStr));
        const double qx = std::stod(trimString(qxStr));
        const double qy = std::stod(trimString(qyStr));
        const double qz = std::stod(trimString(qzStr));
        const double qw = std::stod(trimString(qwStr));

        Eigen::Quaternionf q(static_cast<float>(qw),
                             static_cast<float>(qx),
                             static_cast<float>(qy),
                             static_cast<float>(qz));
        q.normalize();

        Eigen::Isometry3f T_WS = Eigen::Isometry3f::Identity();
        T_WS.linear() = q.toRotationMatrix();
        T_WS.translation() = Eigen::Vector3f(
            static_cast<float>(px),
            static_cast<float>(py),
            static_cast<float>(pz));

        posesByTimestamp[timestamp] = T_WS;
      }

      return !posesByTimestamp.empty();
    }

    bool loadRgbDatasetIndex(
        const std::string& datasetPath,
        const std::string& rgbStream,
        const double imageDelay,
        std::vector<std::pair<uint64_t, std::string>>& rgbFrames)
    {
      const std::string csvPath = datasetPath + "/" + rgbStream + "/data.csv";
      std::ifstream in(csvPath);
      if (!in.good()) {
        LOG(ERROR) << "GS finalization: cannot open RGB index " << csvPath;
        return false;
      }

      std::string line;
      std::getline(in, line); 
      while (std::getline(in, line)) {
        if (line.empty()) {
          continue;
        }
        std::stringstream ss(line);
        std::string tsStr;
        std::string nameStr;
        if (!std::getline(ss, tsStr, ',')) {
          continue;
        }
        if (!std::getline(ss, nameStr, ',')) {
          continue;
        }
        tsStr = trimString(tsStr);
        nameStr = trimString(nameStr);
        if (tsStr.empty() || nameStr.empty()) {
          continue;
        }
        try {

          Time timestamp;
          timestamp.fromNSec(static_cast<uint64_t>(std::stoull(tsStr)));
          const uint64_t correctedTimestamp = static_cast<uint64_t>(
              (timestamp - Duration(imageDelay)).toNSec());
          rgbFrames.emplace_back(correctedTimestamp, nameStr);
        } catch (...) {
          continue;
        }
      }

      std::sort(rgbFrames.begin(),
                rgbFrames.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      return !rgbFrames.empty();
    }

    bool findNearestPose(
        const PoseLookup& posesByTimestamp,
        const uint64_t timestamp,
        Eigen::Isometry3f& T_WS)
    {
      if (posesByTimestamp.empty()) {
        return false;
      }

      auto itAfter = posesByTimestamp.lower_bound(timestamp);
      auto bestIt = posesByTimestamp.end();
      uint64_t bestDt = kRenderPoseToleranceNs + 1;

      if (itAfter != posesByTimestamp.end()) {
        const uint64_t dt = (itAfter->first >= timestamp)
                                ? (itAfter->first - timestamp)
                                : (timestamp - itAfter->first);
        bestDt = dt;
        bestIt = itAfter;
      }
      if (itAfter != posesByTimestamp.begin()) {
        auto itBefore = itAfter;
        --itBefore;
        const uint64_t dt = (itBefore->first >= timestamp)
                                ? (itBefore->first - timestamp)
                                : (timestamp - itBefore->first);
        if (dt < bestDt) {
          bestDt = dt;
          bestIt = itBefore;
        }
      }

      if (bestIt == posesByTimestamp.end() || bestDt > kRenderPoseToleranceNs) {
        return false;
      }

      T_WS = bestIt->second;
      return true;
    }

    enum class GsMappingPoseMatch {
      None,
      ExactTimestamp,
      InterpolatedTimestamp,
      NearestEndpoint,
      StateIdFallback
    };

    bool findMappingPoseAtTimestamp(
        const PoseLookup& posesByTimestamp,
        const uint64_t timestamp,
        Eigen::Isometry3f& T_WS,
        GsMappingPoseMatch& match)
    {
      match = GsMappingPoseMatch::None;
      if (posesByTimestamp.empty()) {
        return false;
      }

      const auto it_after = posesByTimestamp.lower_bound(timestamp);
      if (it_after != posesByTimestamp.end() && it_after->first == timestamp) {
        T_WS = it_after->second;
        match = GsMappingPoseMatch::ExactTimestamp;
        return true;
      }

      if (it_after != posesByTimestamp.begin()
          && it_after != posesByTimestamp.end()) {
        auto it_before = std::prev(it_after);
        const uint64_t dt_before = timestamp - it_before->first;
        const uint64_t dt_after = it_after->first - timestamp;
        const uint64_t bracket_ns = it_after->first - it_before->first;
        if (dt_before <= kGsMappingPoseEndpointToleranceNs
            && dt_after <= kGsMappingPoseEndpointToleranceNs
            && bracket_ns > 0
            && bracket_ns <= kGsMappingPoseMaxBracketNs) {
          const float alpha =
              static_cast<float>(static_cast<double>(dt_before)
                                 / static_cast<double>(bracket_ns));
          Eigen::Quaternionf q_before(it_before->second.linear());
          Eigen::Quaternionf q_after(it_after->second.linear());
          q_before.normalize();
          q_after.normalize();

          T_WS = Eigen::Isometry3f::Identity();
          T_WS.linear() = q_before.slerp(alpha, q_after).normalized().toRotationMatrix();
          T_WS.translation() =
              (1.0f - alpha) * it_before->second.translation()
              + alpha * it_after->second.translation();
          match = GsMappingPoseMatch::InterpolatedTimestamp;
          return true;
        }
      }

      auto nearest = posesByTimestamp.end();
      uint64_t nearest_dt = kGsMappingPoseEndpointToleranceNs + 1;
      if (it_after != posesByTimestamp.end()) {
        nearest = it_after;
        nearest_dt = it_after->first - timestamp;
      }
      if (it_after != posesByTimestamp.begin()) {
        const auto it_before = std::prev(it_after);
        const uint64_t dt_before = timestamp - it_before->first;
        if (dt_before < nearest_dt) {
          nearest = it_before;
          nearest_dt = dt_before;
        }
      }
      if (nearest == posesByTimestamp.end()
          || nearest_dt > kGsMappingPoseEndpointToleranceNs) {
        return false;
      }

      T_WS = nearest->second;
      match = GsMappingPoseMatch::NearestEndpoint;
      return true;
    }

    bool findMappingPoseByTimestampOrState(
        const PoseLookup& posesByStateId,
        const PoseLookup& posesByTimestamp,
        const uint64_t stateId,
        const uint64_t timestamp,
        Eigen::Isometry3f& T_WS,
        GsMappingPoseMatch& match)
    {

      if (findMappingPoseAtTimestamp(
              posesByTimestamp, timestamp, T_WS, match)) {
        return true;
      }
      if (stateId != 0) {
        const auto it_state = posesByStateId.find(stateId);
        if (it_state != posesByStateId.end()) {
          T_WS = it_state->second;
          match = GsMappingPoseMatch::StateIdFallback;
          return true;
        }
      }
      match = GsMappingPoseMatch::None;
      return false;
    }

    void writeSubmapPose(const std::filesystem::path& submapDir,
                         const Eigen::Isometry3f& T_WK)
    {
      std::filesystem::create_directories(submapDir);
      std::ofstream out(submapDir / "T_WK.txt", std::ios::out);
      out << std::setprecision(9);
      const Eigen::Matrix4f M = T_WK.matrix();
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          out << M(r, c);
          if (c < 3) {
            out << " ";
          }
        }
        out << "\n";
      }
    }

    torch::Tensor ensureOpacityShape(torch::Tensor opacity)
    {
      if (!opacity.defined()) {
        return opacity;
      }
      if (opacity.dim() == 1) {
        return opacity.view({-1, 1}).contiguous();
      }
      return opacity.contiguous();
    }


    gs::ErrorDensifyParams toGsErrorDensifyParams(
        const GsDensifyConfig& cfg)
    {
      gs::ErrorDensifyParams out;
      out.enabled = cfg.enabled;
      out.interval = cfg.interval;
      out.tile_size = cfg.tile_size;
      out.top_tiles_ratio = cfg.top_tiles_ratio;
      out.top_peaks_ratio = cfg.top_peaks_ratio;
      out.max_new_gaussians = cfg.max_new_gaussians;
      out.init_scale_mul = cfg.init_scale_mul;
      out.init_z_scale = cfg.init_z_scale;
      out.split_size_ratio = cfg.split_size_ratio;
      out.split_factor = cfg.split_factor;
      out.split_scale_shrink = cfg.split_scale_shrink;
      return out;
    }

    struct GsPoseOptimState {
      torch::Tensor theta;
      torch::Tensor rho;
      std::unique_ptr<torch::optim::Adam> optimizer;
    };

    torch::Tensor skewSymmetric3(const torch::Tensor& v)
    {
      torch::Tensor W = torch::zeros({3, 3}, v.options());
      W.index_put_({0, 1}, -v.index({2}));
      W.index_put_({0, 2}, v.index({1}));
      W.index_put_({1, 0}, v.index({2}));
      W.index_put_({1, 2}, -v.index({0}));
      W.index_put_({2, 0}, -v.index({1}));
      W.index_put_({2, 1}, v.index({0}));
      return W;
    }

    void applyPoseDeltaToCamera(gs::Camera& camera,
                                const torch::Tensor& rho,
                                const torch::Tensor& theta,
                                const torch::Tensor& projRaw)
    {
      if (!rho.defined() || !theta.defined() || rho.numel() != 3 || theta.numel() != 3) {
        return;
      }

      using torch::indexing::Slice;
      torch::NoGradGuard no_grad;
      auto device = camera.T_W2C.device();
      auto dtype = camera.T_W2C.scalar_type();

      auto rho_v = rho.view({3}).to(device, dtype, true).contiguous();
      auto theta_v = theta.view({3}).to(device, dtype, true).contiguous();
      auto I = torch::eye(3, camera.T_W2C.options());
      auto W = skewSymmetric3(theta_v);
      auto W2 = W.mm(W);
      auto angle = torch::norm(theta_v);
      auto angle_safe = torch::clamp(angle, 1.0e-6);

      auto A = torch::sin(angle_safe) / angle_safe;
      auto B = (1.0 - torch::cos(angle_safe)) / (angle_safe * angle_safe);
      auto C = (angle_safe - torch::sin(angle_safe))
               / (angle_safe * angle_safe * angle_safe);

      auto R = I + A * W + B * W2;
      auto V = I + B * W + C * W2;
      auto t = V.mm(rho_v.unsqueeze(1)).squeeze(1);

      auto T_delta = torch::eye(4, camera.T_W2C.options());
      T_delta.index_put_({Slice(0, 3), Slice(0, 3)}, R);
      T_delta.index_put_({Slice(0, 3), 3}, t);

      auto proj_raw_local = projRaw.to(device, dtype, true).contiguous();
      auto updated_T_W2C = camera.T_W2C.mm(T_delta.transpose(0, 1)).contiguous();
      gs::refreshFixedPoseCameraDerivedTensors(camera, updated_T_W2C, proj_raw_local);
    }

    std::vector<std::string> splitCsvColumns(const std::string& line)
    {
      std::vector<std::string> columns;
      std::stringstream ss(line);
      std::string token;
      while (std::getline(ss, token, ',')) {
        columns.push_back(trimString(token));
      }
      return columns;
    }

    std::string joinCsvColumns(const std::vector<std::string>& columns)
    {
      std::ostringstream out;
      for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        out << columns[i];
      }
      return out.str();
    }

    std::string formatScientific(double value)
    {
      std::ostringstream out;
      out << std::scientific << std::setprecision(18) << value;
      return out.str();
    }

    std::filesystem::path buildGsOptimizedTrajectoryPath(
        const std::string& finalTrajectoryCsv)
    {
      std::filesystem::path sourcePath(finalTrajectoryCsv);
      const std::string sourceName = sourcePath.filename().string();
      std::string outputName = sourceName;

      const std::string baSuffix = "-final-ba_trajectory.csv";
      const std::string finalSuffix = "-final_trajectory.csv";
      if (outputName.size() >= baSuffix.size()
          && outputName.compare(outputName.size() - baSuffix.size(),
                                baSuffix.size(),
                                baSuffix)
                 == 0) {
        outputName.replace(outputName.size() - baSuffix.size(),
                           baSuffix.size(),
                           "-final-ba-gs_trajectory.csv");
      } else if (outputName.size() >= finalSuffix.size()
                 && outputName.compare(outputName.size() - finalSuffix.size(),
                                       finalSuffix.size(),
                                       finalSuffix)
                        == 0) {
        outputName.replace(outputName.size() - finalSuffix.size(),
                           finalSuffix.size(),
                           "-final-gs_trajectory.csv");
      } else {
        outputName += "-gs.csv";
      }
      return sourcePath.parent_path() / outputName;
    }

    bool writeGsOptimizedTrajectoryCsv(const std::string& sourceCsv,
                                       const PoseLookup& optimizedPosesByTimestamp,
                                       const std::filesystem::path& outputCsv,
                                       size_t* replacedRows = nullptr)
    {
      std::ifstream in(sourceCsv);
      if (!in.good()) {
        return false;
      }
      std::ofstream out(outputCsv, std::ios::out | std::ios::trunc);
      if (!out.good()) {
        return false;
      }

      std::string line;
      if (!std::getline(in, line)) {
        return false;
      }
      out << line << "\n";

      size_t replaced = 0;
      while (std::getline(in, line)) {
        if (line.empty()) {
          out << "\n";
          continue;
        }
        auto columns = splitCsvColumns(line);
        if (columns.size() >= 8) {
          try {
            const uint64_t timestamp = static_cast<uint64_t>(std::stoull(columns[0]));
            const auto poseIt = optimizedPosesByTimestamp.find(timestamp);
            if (poseIt != optimizedPosesByTimestamp.end()) {
              const Eigen::Isometry3f& T_WS = poseIt->second;
              Eigen::Quaternionf q(T_WS.linear());
              q.normalize();

              columns[1] = formatScientific(static_cast<double>(T_WS.translation().x()));
              columns[2] = formatScientific(static_cast<double>(T_WS.translation().y()));
              columns[3] = formatScientific(static_cast<double>(T_WS.translation().z()));
              columns[4] = formatScientific(static_cast<double>(q.x()));
              columns[5] = formatScientific(static_cast<double>(q.y()));
              columns[6] = formatScientific(static_cast<double>(q.z()));
              columns[7] = formatScientific(static_cast<double>(q.w()));
              replaced++;
            }
          } catch (...) {
          }
        }
        out << joinCsvColumns(columns) << "\n";
      }

      if (replacedRows) {
        *replacedRows = replaced;
      }
      return true;
    }

    torch::Tensor transformPointsToWorld(const torch::Tensor& xyzK,
                                         const Eigen::Isometry3f& T_WK)
    {
      torch::NoGradGuard no_grad;
      auto xyz = xyzK.to(torch::kCPU).to(torch::kFloat32).contiguous();

      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rrm = T_WK.linear();
      Eigen::Vector3f t = T_WK.translation();

      auto R = torch::from_blob((void*) Rrm.data(), {3, 3}, torch::kFloat32).clone();
      auto tt = torch::from_blob((void*) t.data(), {3}, torch::kFloat32).clone();
      return xyz.matmul(R.t()) + tt;
    }

    torch::Tensor transformRotWXYZToWorld(const torch::Tensor& rotK,
                                          const Eigen::Isometry3f& T_WK)
    {
      torch::NoGradGuard no_grad;
      auto rot = rotK.to(torch::kCPU).to(torch::kFloat32).contiguous();
      if (!rot.defined() || rot.numel() == 0) {
        return rot;
      }
      auto out = torch::empty_like(rot);
      const auto* inPtr = rot.data_ptr<float>();
      auto* outPtr = out.data_ptr<float>();

      const Eigen::Quaternionf q_WK(T_WK.linear());
      const int64_t N = rot.size(0);
      for (int64_t i = 0; i < N; ++i) {
        const Eigen::Quaternionf q_K(
            inPtr[4 * i + 0], inPtr[4 * i + 1], inPtr[4 * i + 2], inPtr[4 * i + 3]);
        Eigen::Quaternionf q_W = q_WK * q_K;
        q_W.normalize();
        outPtr[4 * i + 0] = q_W.w();
        outPtr[4 * i + 1] = q_W.x();
        outPtr[4 * i + 2] = q_W.y();
        outPtr[4 * i + 3] = q_W.z();
      }
      return out;
    }

    torch::Tensor transformPointsToSubmap(const torch::Tensor& xyzW,
                                          const Eigen::Isometry3f& T_WK)
    {
      torch::NoGradGuard no_grad;
      auto xyz = xyzW.to(torch::kCPU).to(torch::kFloat32).contiguous();

      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rrm = T_WK.linear();
      Eigen::Vector3f t = T_WK.translation();

      auto R = torch::from_blob((void*) Rrm.data(), {3, 3}, torch::kFloat32).clone();
      auto tt = torch::from_blob((void*) t.data(), {3}, torch::kFloat32).clone();
      return (xyz - tt).matmul(R);
    }

    torch::Tensor transformRotWXYZToSubmap(const torch::Tensor& rotW,
                                           const Eigen::Isometry3f& T_WK)
    {
      torch::NoGradGuard no_grad;
      auto rot = rotW.to(torch::kCPU).to(torch::kFloat32).contiguous();
      if (!rot.defined() || rot.numel() == 0) {
        return rot;
      }
      auto out = torch::empty_like(rot);
      const auto* inPtr = rot.data_ptr<float>();
      auto* outPtr = out.data_ptr<float>();

      const Eigen::Quaternionf q_WK(T_WK.linear());
      const Eigen::Quaternionf q_KW = q_WK.conjugate();
      const int64_t N = rot.size(0);
      for (int64_t i = 0; i < N; ++i) {
        const Eigen::Quaternionf q_W(
            inPtr[4 * i + 0], inPtr[4 * i + 1], inPtr[4 * i + 2], inPtr[4 * i + 3]);
        Eigen::Quaternionf q_K = q_KW * q_W;
        q_K.normalize();
        outPtr[4 * i + 0] = q_K.w();
        outPtr[4 * i + 1] = q_K.x();
        outPtr[4 * i + 2] = q_K.y();
        outPtr[4 * i + 3] = q_K.z();
      }
      return out;
    }

    bool gsTensorDefinedNonEmpty(const torch::Tensor& tensor)
    {
      return tensor.defined() && tensor.numel() > 0;
    }

    bool gsCpuSnapshotHasValidXyz(const gs::PlyTensorsCPU& snapshot)
    {
      return gsTensorDefinedNonEmpty(snapshot.xyz);
    }

    bool gsModelHasValidXyz(const std::unique_ptr<gs::GaussianModel>& model)
    {
      if (!model) {
        return false;
      }
      const torch::Tensor xyz = model->Get_xyz();
      return gsTensorDefinedNonEmpty(xyz);
    }
  } 

  struct SubmappingInterface::GsRuntime {
    struct GsSamplePoseMetadata {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      uint64_t timestamp_ns = 0;
      uint64_t state_id = 0;
      int cam_idx = -1;
      Eigen::Matrix4f T_SC = Eigen::Matrix4f::Identity();
    };

    struct GsSubmapState {
      std::unique_ptr<gs::GaussianModel> model;
      std::vector<gs::Camera> gsCamList;
      std::vector<torch::Tensor> gtImgList;
      std::vector<gs::GeometryTarget> geometryTargetList;
      std::vector<gs::Camera> gsCamListCpu;
      std::vector<torch::Tensor> gtImgListCpu;
      std::vector<torch::Tensor> depthTargetListCpu;
      gs::PlyTensorsCPU cpuModelSnapshot;
      std::filesystem::path outputDirectory;
      size_t creationIndex = 0;
      bool hasCpuModelSnapshot = false;
      bool offloadQueued = false;
      SubmappingInterface::GsFisherDiag fisherHistDiag;
      size_t fisherHistViewCount = 0;
      size_t fisherEstimateUpdateCount = 0;
      size_t fisherCacheVersion = 0;
      std::vector<GsSamplePoseMetadata, Eigen::aligned_allocator<GsSamplePoseMetadata>>
          poseMetadataCpu;
    };

    gs::param::OptimizationParameters optimParams;
    std::filesystem::path outputDirectory;
    gs::DataQueue dataQueue;
    std::unordered_map<uint64_t, GsSubmapState> submapStates;
    std::unique_ptr<gs::GaussianModel> mergedModel;
    size_t nextSubmapCreationIndex = 0;
  };

namespace {
    constexpr size_t kOccupancyPreviewQueueSize = 1;

    bool newestDepthMeasurementTimestamp(const DepthMeasurementPtr& measurement,
                                         okvis::Time& newest_timestamp)
    {
      if (!measurement || measurement->empty()) {
        return false;
      }

      bool found = false;
      for (const auto& camera_measurements : *measurement) {
        for (const auto& camera_measurement : camera_measurements.second) {
          if (!found || camera_measurement.timeStamp > newest_timestamp) {
            newest_timestamp = camera_measurement.timeStamp;
            found = true;
          }
        }
      }
      return found;
    }

    bool depthMeasurementReadyForState(const DepthMeasurementPtr& measurement,
                                       const size_t mapping_camera_id,
                                       const okvis::Time& state_timestamp)
    {
      if (!measurement) {
        return false;
      }
      const auto mapping_camera_it = measurement->find(mapping_camera_id);
      if (mapping_camera_it == measurement->end() || mapping_camera_it->second.empty()) {
        return false;
      }

      okvis::Time newest_timestamp;
      return newestDepthMeasurementTimestamp(measurement, newest_timestamp)
             && newest_timestamp <= state_timestamp;
    }

    void recordLatestWinsDrops(const char* stage,
                               std::atomic<uint64_t>& counter,
                               const size_t dropped)
    {
      if (dropped == 0) {
        return;
      }
      const uint64_t previous = counter.fetch_add(dropped, std::memory_order_relaxed);
      const uint64_t total = previous + dropped;
      if (previous < 5 || previous / 100 != total / 100) {
        LOG(INFO) << "Realtime latest-wins dropped old " << stage
                  << ": batch=" << dropped << " total=" << total;
      }
    }

  }

  void get_bbox(const se::Submap<okvis::SupereightMapType>& submap,
                const Eigen::Matrix4f& T_AB,
                Eigen::Vector3f& bbox_world_max,
                Eigen::Vector3f& bbox_world_min)
  {

    Eigen::Vector3f map_min = submap.map->aabb().min();
    Eigen::Vector3f map_max = submap.map->aabb().max();

    Eigen::Vector3f corner000(map_min.x(), map_min.y(), map_min.z());
    Eigen::Vector3f corner100(map_max.x(), map_min.y(), map_min.z());
    Eigen::Vector3f corner010(map_min.x(), map_max.y(), map_min.z());
    Eigen::Vector3f corner001(map_min.x(), map_min.y(), map_max.z());
    Eigen::Vector3f corner110(map_max.x(), map_max.y(), map_min.z());
    Eigen::Vector3f corner101(map_max.x(), map_min.y(), map_max.z());
    Eigen::Vector3f corner011(map_min.x(), map_max.y(), map_max.z());
    Eigen::Vector3f corner111(map_max.x(), map_max.y(), map_max.z());

    corner000 = (T_AB * corner000.homogeneous()).head<3>();
    corner100 = (T_AB * corner100.homogeneous()).head<3>();
    corner010 = (T_AB * corner010.homogeneous()).head<3>();
    corner001 = (T_AB * corner001.homogeneous()).head<3>();
    corner110 = (T_AB * corner110.homogeneous()).head<3>();
    corner101 = (T_AB * corner101.homogeneous()).head<3>();
    corner011 = (T_AB * corner011.homogeneous()).head<3>();
    corner111 = (T_AB * corner111.homogeneous()).head<3>();

    float x_min = std::min({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});
    float x_max = std::max({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});

    float y_min = std::min({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});
    float y_max = std::max({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});

    float z_min = std::min({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});
    float z_max = std::max({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});

    bbox_world_min.x() = x_min;
    bbox_world_min.y() = y_min;
    bbox_world_min.z() = z_min;
    bbox_world_max.x() = x_max;
    bbox_world_max.y() = y_max;
    bbox_world_max.z() = z_max;

  }

  bool SubmappingInterface::publishGsPreview(GsPreviewFrame& previewFrame)
  {
    if(gsPreviewQueue_.Empty()) {
      return false;
    }

    gsPreviewQueue_.PopNonBlocking(&previewFrame);
    return true;
  }

  void SubmappingInterface::runGsPreviewWorker(std::shared_ptr<GsRuntime> runtime)
  {
    if (!runtime) return;
    gs::DataPacket packet;
    while (runtime->dataQueue.pop(&packet)) {
      try {
        if (packet.rendered_rgb.empty() && packet.rendered_rgb_staging.valid()) {
          packet.rendered_rgb = gs::consumePreviewRgbCpu(std::move(packet.rendered_rgb_staging));
        }
        auto preview = toGsPreviewFrame(
            std::move(packet), gsPreviewSubmapCount_.load(std::memory_order_relaxed));
        gsPreviewQueue_.PushNonBlockingDroppingIfFull(std::move(preview), kGsPreviewQueueSize);
      } catch (const std::exception& e) {
        LOG(WARNING) << "GS preview conversion dropped a frame: " << e.what();
      }
    }
  }

  void SubmappingInterface::stopGsPreviewWorker()
  {
    if (gsPreviewRuntime_) {
      gsPreviewRuntime_->dataQueue.shutdown();
    }
    if (gsPreviewWorker_.joinable()) {
      gsPreviewWorker_.join();
    }
    gsPreviewRuntime_.reset();
  }

  bool SubmappingInterface::publishOccupancyPreview(OccupancyPreviewFrame& previewFrame)
  {
    if (occupancyPreviewQueue_.Empty()) {
      return false;
    }
    occupancyPreviewQueue_.PopNonBlocking(&previewFrame);
    return true;
  }

  bool SubmappingInterface::buildOccupancyPreviewSnapshot(OccupancyPreviewFrame& previewFrame,
                                                          const bool refresh_geometry)
  {
    if (!occupancyViewerConfig_.enable) {
      return false;
    }
    if (seSubmapLookup_.empty()) {
      return false;
    }

    const int max_points = occupancyViewerConfig_.max_points;
    const int stride = occupancyViewerConfig_.voxel_stride;
    const int num_submaps = std::max(1, static_cast<int>(seSubmapLookup_.size()));
    const int per_submap_occ_budget = std::max(1, max_points / num_submaps);

    for (auto it = occupancySubmapCaches_.begin(); it != occupancySubmapCaches_.end();) {
      if (seSubmapLookup_.find(it->first) == seSubmapLookup_.end()) {
        occupancyDirtySubmaps_.erase(it->first);
        it = occupancySubmapCaches_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = occupancyDirtySubmaps_.begin(); it != occupancyDirtySubmaps_.end();) {
      if (seSubmapLookup_.find(*it) == seSubmapLookup_.end()) {
        it = occupancyDirtySubmaps_.erase(it);
      } else {
        ++it;
      }
    }

    std::unordered_set<uint64_t> geometry_updated_submaps;
    geometry_updated_submaps.reserve(seSubmapLookup_.size());
    bool has_any_map = false;
    for (const auto& submap_kv : seSubmapLookup_) {
      const uint64_t submap_id = submap_kv.first;
      const auto& submap = submap_kv.second;
      if (!submap.map) {
        continue;
      }
      has_any_map = true;

      auto cache_it = occupancySubmapCaches_.find(submap_id);
      const bool cache_missing = (cache_it == occupancySubmapCaches_.end());
      if (cache_missing) {
        cache_it = occupancySubmapCaches_.emplace(submap_id, OccupancySubmapCache{}).first;
      }
      auto& cache = cache_it->second;
      const bool cache_mismatch =
          (cache.cached_stride != stride || cache.cached_occ_budget != per_submap_occ_budget);
      if (refresh_geometry || cache_missing || cache_mismatch || cache.needs_full_rebuild) {
        if (rebuildOccupancySubmapCache(submap_id,
                                        submap,
                                        stride,
                                        per_submap_occ_budget)) {
          geometry_updated_submaps.insert(submap_id);
          occupancyDirtySubmaps_.erase(submap_id);
        }
      }
    }
    if (!has_any_map) {
      return false;
    }

    if (refresh_geometry) {
      gsFisherVisualizationRefreshRequested_.store(true);
      gsGlobalFisherVisualizationRefreshRequested_.store(true);
      updateGsActiveFisherVisualization();
      updateGsGlobalFisherVisualization();
    }

    OccupancyPreviewFrame snapshot_frame;
    auto submap_clouds =
        std::make_shared<std::vector<OccupancyPreviewFrame::SubmapCloud>>();
    submap_clouds->reserve(seSubmapLookup_.size());
    size_t total_occ_points = 0;
    for (const auto& submap_kv : seSubmapLookup_) {
      const uint64_t submap_id = submap_kv.first;
      const auto& submap = submap_kv.second;
      if (!submap.map) {
        continue;
      }
      auto cache_it = occupancySubmapCaches_.find(submap_id);
      if (cache_it == occupancySubmapCaches_.end()) {
        continue;
      }
      OccupancyPreviewFrame::SubmapCloud submap_cloud;
      submap_cloud.submap_id = submap_id;
      submap_cloud.occupied_points_local = cache_it->second.occupied_points_local;
      submap_cloud.occupied_colours = cache_it->second.occupied_colours;
      submap_cloud.occupied_fisher_colours = cache_it->second.occupied_fisher_colours;
      submap_cloud.occupied_global_fisher_colours =
          cache_it->second.occupied_global_fisher_colours;
      submap_cloud.T_WK = submap.T_WK.matrix();
      submap_cloud.num_points = cache_it->second.num_points;
      submap_cloud.geometry_revision = cache_it->second.geometry_revision;
      submap_cloud.colour_revision = cache_it->second.colour_revision;
      submap_cloud.geometry_updated = geometry_updated_submaps.count(submap_id) > 0;
      total_occ_points += submap_cloud.num_points;
      submap_clouds->push_back(std::move(submap_cloud));
    }
    snapshot_frame.submap_clouds = submap_clouds;
    {
      std::lock_guard lockCurrentPose(currentPose_mtx_);
      snapshot_frame.T_WS = currentPose_;
    }
    snapshot_frame.num_submaps = seSubmapLookup_.size();
    snapshot_frame.num_points = total_occ_points;

    snapshot_frame.integration_counter = integration_counter_;
    previewFrame = std::move(snapshot_frame);
    return true;
  }

  void SubmappingInterface::setGsDensifyConfig(const GsDensifyConfig& config)
  {
    gsDensifyConfig_ = config;
    if (!gsRuntime_) {
      return;
    }

    const auto densify_params = toGsErrorDensifyParams(gsDensifyConfig_);
    for (auto& kv : gsRuntime_->submapStates) {
      if (kv.second.model) {
        kv.second.model->densify_params = densify_params;
      }
    }
    if (gsRuntime_->mergedModel) {
      gsRuntime_->mergedModel->densify_params = densify_params;
    }
  }

  void SubmappingInterface::setOccupancyViewerConfig(const OccupancyViewerConfig& config)
  {
    occupancyViewerConfig_ = config;
    occupancyViewerConfig_.update_interval = std::max(1, occupancyViewerConfig_.update_interval);
    occupancyViewerConfig_.voxel_stride = std::max(1, occupancyViewerConfig_.voxel_stride);
    occupancyViewerConfig_.max_points = std::max(1000, occupancyViewerConfig_.max_points);
    occupancyViewerConfig_.occupied_point_size = std::max(1, occupancyViewerConfig_.occupied_point_size);

    occupancyViewerConfig_.max_fps = std::max(1, occupancyViewerConfig_.max_fps);
    occupancyViewerConfig_.window_width = std::max(1, occupancyViewerConfig_.window_width);
    occupancyViewerConfig_.window_height = std::max(1, occupancyViewerConfig_.window_height);

    occupancyViewerConfig_.fisher_update_interval =
        std::max(1, occupancyViewerConfig_.fisher_update_interval);
    occupancyViewerConfig_.fisher_display_regularizer =
        std::max(1.0e-6f, occupancyViewerConfig_.fisher_display_regularizer);
    occupancyViewerConfig_.fisher_eig_regularizer =
        std::max(1.0e-12f, occupancyViewerConfig_.fisher_eig_regularizer);
  }

  void SubmappingInterface::shutdown()
  {
    std::lock_guard<std::mutex> shutdown_lock(shutdownMutex_);
    if (shutdownComplete_) return;
    {
      std::lock_guard<std::mutex> lock(finishMutex_);
      isFinished_ = true;
    }
    depthMeasurements_.Shutdown();
    lidarMeasurements_.Shutdown();
    stateUpdates_.Shutdown();
    supereightFrames_.Shutdown();
    occupancyPreviewQueue_.Shutdown();
    gsPreviewQueue_.Shutdown();
    stopGsPreviewWorker();
    if (dataIntegration_.joinable()) dataIntegration_.join();
    if (submapIntegration_.joinable()) submapIntegration_.join();
    waitForSubmapFinalization();
    {
      std::lock_guard<std::mutex> lock(submapFinalizeMutex_);
      submapFinalizeStop_ = true;
    }
    submapFinalizeCv_.notify_all();
    waitForGsModelOffloads();
    {
      std::lock_guard<std::mutex> lock(gsModelOffloadMutex_);
      gsModelOffloadStop_ = true;
    }
    if (submapFinalizeWorker_.joinable()) submapFinalizeWorker_.join();
    shutdownComplete_ = true;
  }

  void SubmappingInterface::setGsFisherVisualizationEnabled(const bool enabled)
  {
    if (enabled) setGsGlobalFisherVisualizationEnabled(false);
    if (gsFisherVisualizationEnabled_.exchange(enabled) != enabled) {
      gsFisherVisualizationRefreshRequested_.store(true);
    }
  }

  void SubmappingInterface::setGsGlobalFisherVisualizationEnabled(const bool enabled)
  {
    if (enabled) setGsFisherVisualizationEnabled(false);
    if (gsGlobalFisherVisualizationEnabled_.exchange(enabled) != enabled) {
      gsGlobalFisherVisualizationRefreshRequested_.store(true);
    }
  }

  void SubmappingInterface::markOccupancySubmapDirty(
      const uint64_t submap_id,
      const std::unordered_set<const se::OctantBase*>* const updated_octants)
  {
    if (submap_id == UNINITIALIZED_ID) {
      return;
    }

    occupancyDirtySubmaps_.insert(submap_id);
    auto& cache = occupancySubmapCaches_[submap_id];
    if (!updated_octants || updated_octants->empty()) {
      cache.needs_full_rebuild = true;
      cache.dirty_block_queue.clear();
      cache.dirty_block_set.clear();
      return;
    }

    bool saw_block = false;
    bool requires_full_rebuild = false;
    for (const se::OctantBase* octant_ptr : *updated_octants) {
      if (!octant_ptr) {
        continue;
      }
      if (!octant_ptr->is_block) {
        requires_full_rebuild = true;
        continue;
      }
      saw_block = true;
      const OccupancyBlockKey block_key{
          octant_ptr->coord.x(), octant_ptr->coord.y(), octant_ptr->coord.z()};
      if (cache.dirty_block_set.insert(block_key).second) {
        cache.dirty_block_queue.push_back(block_key);
      }
    }

    if (requires_full_rebuild || !saw_block) {
      cache.needs_full_rebuild = true;
      cache.dirty_block_queue.clear();
      cache.dirty_block_set.clear();
    }
  }

  void SubmappingInterface::sampleOccupancyBlockFromPointer(
      const se::Submap<SupereightMapType>& submap,
      const SupereightMapType::BlockType* const block_ptr,
      const int stride,
      OccupancyTrackedBlock& tracked_block) const
  {
    tracked_block.active = true;
    tracked_block.occupied_points_local.clear();
    tracked_block.occupied_colours.clear();
    if (!submap.map || !block_ptr) {
      tracked_block.active = false;
      return;
    }

    const int block_scale = std::max(0, block_ptr->current_scale);
    const int voxel_size = 1 << block_scale;
    const int voxel_step = std::max(1, voxel_size * stride);

    for (int z = 0; z < SupereightMapType::BlockType::size; z += voxel_step) {
      for (int y = 0; y < SupereightMapType::BlockType::size; y += voxel_step) {
        for (int x = 0; x < SupereightMapType::BlockType::size; x += voxel_step) {
          const Eigen::Vector3i voxel_coord = block_ptr->coord + Eigen::Vector3i(x, y, z);
          const auto& voxel_data = block_ptr->data(voxel_coord, block_scale);
          const bool occupied = voxel_data.field.observed && se::get_field(voxel_data) > 0.0f;
          if (!occupied) {
            continue;
          }

          Eigen::Vector3f point_K;
          submap.map->voxelToPoint(voxel_coord, voxel_size, point_K);
          tracked_block.occupied_points_local.push_back(point_K);
          tracked_block.occupied_colours.push_back(getVoxelColour(voxel_data));
        }
      }
    }

  }

  bool SubmappingInterface::sampleOccupancyBlock(
      const se::Submap<SupereightMapType>& submap,
      const OccupancyBlockKey& block_key,
      const int stride,
      OccupancyTrackedBlock& tracked_block) const
  {
    if (!submap.map) {
      return false;
    }
    const Eigen::Vector3i block_coord(block_key.x, block_key.y, block_key.z);
    const auto& octree = submap.map->getOctree();
    const auto* octant_ptr = se::fetcher::block<typename SupereightMapType::OctreeType>(
        block_coord, octree.getRoot());
    if (!octant_ptr || !octant_ptr->is_block) {
      return false;
    }
    tracked_block.key = block_key;
    sampleOccupancyBlockFromPointer(
        submap, static_cast<const SupereightMapType::BlockType*>(octant_ptr), stride,
        tracked_block);
    return tracked_block.active;
  }

  void SubmappingInterface::rebuildOccupancySubmapPoints(const uint64_t submap_id,
                                                         OccupancySubmapCache& cache,
                                                         const int occ_budget)
  {
    auto occupied_points = std::make_shared<OccupancyPreviewFrame::PointVector>();
    auto occupied_colours = std::make_shared<OccupancyPreviewFrame::ColourVector>();

    const size_t occ_budget_sz = static_cast<size_t>(std::max(occ_budget, 0));
    occupied_points->reserve(occ_budget_sz);
    occupied_colours->reserve(occ_budget_sz);

    for (const auto& tracked_block : cache.tracked_blocks) {
      if (!tracked_block.active) {
        continue;
      }

      if (occupied_points->size() < occ_budget_sz) {
        const size_t remaining_occ = occ_budget_sz - occupied_points->size();
        const size_t take_occ = std::min(remaining_occ, tracked_block.occupied_points_local.size());
        occupied_points->insert(occupied_points->end(),
                                tracked_block.occupied_points_local.begin(),
                                tracked_block.occupied_points_local.begin() + take_occ);
        occupied_colours->insert(occupied_colours->end(),
                                 tracked_block.occupied_colours.begin(),
                                 tracked_block.occupied_colours.begin() + take_occ);
      }

      if (occupied_points->size() >= occ_budget_sz) {
        break;
      }
    }

    cache.occupied_points_local = occupied_points;
    cache.occupied_colours = occupied_colours;
    cache.num_points = occupied_points->size();
    recolorOccupancySubmapFromFisher(cache);
    recolorOccupancySubmapFromGlobalFisher(submap_id, cache);
  }

  void SubmappingInterface::recolorOccupancySubmapFromFisher(OccupancySubmapCache& cache) const
  {
    if (!cache.occupied_points_local || cache.occupied_points_local->empty()
        || cache.fisher_score_grid.empty() || cache.fisher_cell_size <= 1.0e-6f) {
      cache.occupied_fisher_colours.reset();
      return;
    }

    auto fisher_colours = std::make_shared<OccupancyPreviewFrame::ColourVector>();
    fisher_colours->reserve(cache.occupied_points_local->size());
    const auto makeKey = [&](const Eigen::Vector3f& p) {
      return OccupancyBlockKey{static_cast<int>(std::floor(p.x() / cache.fisher_cell_size)),
                               static_cast<int>(std::floor(p.y() / cache.fisher_cell_size)),
                               static_cast<int>(std::floor(p.z() / cache.fisher_cell_size))};
    };

    for (const auto& p : *cache.occupied_points_local) {
      const OccupancyBlockKey key = makeKey(p);
      float best_score = 0.0f;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const OccupancyBlockKey neighbor{key.x + dx, key.y + dy, key.z + dz};
            const auto it = cache.fisher_score_grid.find(neighbor);
            if (it != cache.fisher_score_grid.end()) {
              best_score = std::max(best_score, it->second);
            }
          }
        }
      }
      fisher_colours->push_back(fisherHeatRgb(best_score));
    }
    cache.occupied_fisher_colours = fisher_colours;
  }

  void SubmappingInterface::recolorOccupancySubmapFromGlobalFisher(
      const uint64_t submap_id, OccupancySubmapCache& cache) const
  {
    if (!cache.occupied_points_local || cache.occupied_points_local->empty()
        || globalFisherDisplayGrid_.empty() || globalFisherCellSize_ <= 1.0e-6f) {
      cache.occupied_global_fisher_colours.reset();
      return;
    }

    const auto submap_it = seSubmapLookup_.find(submap_id);
    if (submap_it == seSubmapLookup_.end()) {
      cache.occupied_global_fisher_colours.reset();
      return;
    }

    auto fisher_colours = std::make_shared<OccupancyPreviewFrame::ColourVector>();
    fisher_colours->reserve(cache.occupied_points_local->size());
    const Eigen::Matrix4f& T_WK = submap_it->second.T_WK.matrix();
    const auto makeKey = [&](const Eigen::Vector3f& p_world) {
      return OccupancyBlockKey{
          static_cast<int>(std::floor(p_world.x() / globalFisherCellSize_)),
          static_cast<int>(std::floor(p_world.y() / globalFisherCellSize_)),
          static_cast<int>(std::floor(p_world.z() / globalFisherCellSize_))};
    };

    for (const auto& p_local : *cache.occupied_points_local) {
      const Eigen::Vector3f p_world = (T_WK * p_local.homogeneous()).head<3>();
      const OccupancyBlockKey key = makeKey(p_world);
      float best_score = 0.0f;
      bool have_score = false;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const OccupancyBlockKey neighbor{key.x + dx, key.y + dy, key.z + dz};
            const auto it = globalFisherDisplayGrid_.find(neighbor);
            if (it != globalFisherDisplayGrid_.end()) {
              best_score = std::max(best_score, it->second);
              have_score = true;
            }
          }
        }
      }
      if (have_score) {
        fisher_colours->push_back(fisherHeatRgb(best_score));
      } else {
        fisher_colours->push_back(Eigen::Vector3i(70, 70, 70));
      }
    }
    cache.occupied_global_fisher_colours = fisher_colours;
  }

  bool SubmappingInterface::rebuildOccupancySubmapCache(const uint64_t submap_id,
                                                        const se::Submap<SupereightMapType>& submap,
                                                        const int stride,
                                                        const int occ_budget)
  {
    if (!submap.map) {
      return false;
    }

    auto& cache = occupancySubmapCaches_[submap_id];
    cache.tracked_blocks.clear();
    cache.tracked_block_indices.clear();
    cache.dirty_block_queue.clear();
    cache.dirty_block_set.clear();
    cache.needs_full_rebuild = false;

    const auto& octree = submap.map->getOctree();
    for (auto octant_it = octree.begin(); octant_it != octree.end(); ++octant_it) {
      const auto* octant_ptr = *octant_it;
      if (!octant_ptr || !octant_ptr->is_block) {
        continue;
      }

      OccupancyTrackedBlock tracked_block;
      tracked_block.key = OccupancyBlockKey{
          octant_ptr->coord.x(), octant_ptr->coord.y(), octant_ptr->coord.z()};
      sampleOccupancyBlockFromPointer(
          submap, static_cast<const SupereightMapType::BlockType*>(octant_ptr), stride,
          tracked_block);
      if (!tracked_block.active) {
        continue;
      }

      cache.tracked_block_indices.emplace(tracked_block.key, cache.tracked_blocks.size());
      cache.tracked_blocks.push_back(std::move(tracked_block));

    }

    rebuildOccupancySubmapPoints(submap_id, cache, occ_budget);
    cache.geometry_revision++;
    cache.cached_stride = stride;
    cache.cached_occ_budget = occ_budget;
    return true;
  }

  SubmappingInterface::GsFisherDiag SubmappingInterface::regularizeInvertGsFisherDiag(
      const GsFisherDiag& fisher_diag, const float lambda) const
  {
    const float safe_lambda = std::max(1.0e-12f, lambda);
    auto invert_group = [&](const torch::Tensor& fisher_group) {
      if (!fisher_group.defined() || fisher_group.numel() == 0) {
        return torch::Tensor();
      }
      return torch::reciprocal(fisher_group + safe_lambda);
    };

    GsFisherDiag out;
    out.xyz = invert_group(fisher_diag.xyz);
    out.opacity = invert_group(fisher_diag.opacity);
    return out;
  }

  SubmappingInterface::GsFisherDiag SubmappingInterface::addGsFisherDiag(
      const GsFisherDiag& lhs, const GsFisherDiag& rhs) const
  {
    auto add_group = [&](const torch::Tensor& lhs_group, const torch::Tensor& rhs_group) {
      if (lhs_group.defined() && lhs_group.numel() > 0) {
        if (rhs_group.defined() && rhs_group.numel() > 0) {
          return lhs_group + rhs_group;
        }
        return lhs_group.clone();
      }
      if (rhs_group.defined() && rhs_group.numel() > 0) {
        return rhs_group.clone();
      }
      return torch::Tensor();
    };

    GsFisherDiag out;
    out.xyz = add_group(lhs.xyz, rhs.xyz);
    out.opacity = add_group(lhs.opacity, rhs.opacity);
    return out;
  }

  void SubmappingInterface::invalidateGsSubmapFisherCache(const uint64_t submap_id)
  {
    if (!gsRuntime_) {
      return;
    }
    const auto state_it = gsRuntime_->submapStates.find(submap_id);
    if (state_it == gsRuntime_->submapStates.end()) {
      return;
    }

    auto& gs_state = state_it->second;
    gs_state.fisherHistDiag = GsFisherDiag{};
    gs_state.fisherHistViewCount = 0;
    gs_state.fisherEstimateUpdateCount = 0;
    gs_state.fisherCacheVersion++;
  }

  torch::Tensor SubmappingInterface::gsSubmapXyzCpu(const uint64_t submap_id)
  {
    torch::Tensor xyz;
    {
      std::lock_guard<std::mutex> lock(gsModelOffloadMutex_);
      const auto state_it = gsRuntime_->submapStates.find(submap_id);
      if (state_it == gsRuntime_->submapStates.end()) return {};
      const auto& state = state_it->second;
      if (state.offloadQueued) return {};
      if (gsModelHasValidXyz(state.model)) {
        xyz = state.model->Get_xyz();
      } else if (state.hasCpuModelSnapshot) {
        xyz = state.cpuModelSnapshot.xyz;
      }
    }
    if (!gsTensorDefinedNonEmpty(xyz)) return {};
    return xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  }

  bool SubmappingInterface::ensureGsSubmapModelLoaded(const uint64_t submap_id)
  {
    if (!gsRuntime_) {
      return false;
    }
    const auto state_it = gsRuntime_->submapStates.find(submap_id);
    if (state_it == gsRuntime_->submapStates.end()) {
      return false;
    }

    auto& gs_state = state_it->second;
    if (gsModelHasValidXyz(gs_state.model)) {
      return true;
    }
    if (gs_state.offloadQueued) {
      waitForGsModelOffloads();
      if (gsModelHasValidXyz(gs_state.model)) {
        return true;
      }
    }
    if (!gs_state.hasCpuModelSnapshot || !gsCpuSnapshotHasValidXyz(gs_state.cpuModelSnapshot)) {
      return false;
    }

    if (!gs_state.outputDirectory.empty()) {
      std::filesystem::create_directories(gs_state.outputDirectory);
    }
    try {
      if (!gs_state.model) {
        gs_state.model = std::make_unique<gs::GaussianModel>(
            gsRuntime_->optimParams, gs_state.outputDirectory.string());
      }
      gs_state.model->densify_params = toGsErrorDensifyParams(gsDensifyConfig_);
      gs_state.model->Import_ply_tensors_cpu(gs_state.cpuModelSnapshot);
      if (gsModelHasValidXyz(gs_state.model)) {
        return true;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to restore GS model for submap " << submap_id
                   << " from CPU snapshot: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to restore GS model for submap " << submap_id
                   << " from CPU snapshot with unknown exception.";
    }

    gs_state.model.reset();
    return false;
  }

  void SubmappingInterface::updateGsSubmapFisherFromTrainingSamples(
      const uint64_t submap_id,
      const gs::FisherTrainingSampleVector& fisher_training_samples)
  {
    if (!gsRuntime_ || fisher_training_samples.empty() || !ensureGsSubmapModelLoaded(submap_id)) {
      return;
    }

    const auto state_it = gsRuntime_->submapStates.find(submap_id);
    if (state_it == gsRuntime_->submapStates.end() || !state_it->second.model) {
      return;
    }

    auto& gs_state = state_it->second;
    auto& model = *gs_state.model;
    const auto target_xyz = model.Get_xyz();
    const auto target_opacity = model.Get_opacity_raw();
    const float ema_alpha = kGsFisherHistoryEmaAlpha;

    const auto pad_group = [](torch::Tensor& group, const torch::Tensor& reference) {
      if (!reference.defined() || reference.numel() == 0) {
        group = torch::Tensor();
        return;
      }
      if (!group.defined() || group.numel() == 0) {
        group = torch::zeros_like(reference);
        return;
      }
      if (group.sizes().equals(reference.sizes())) {
        return;
      }
      if (group.dim() == reference.dim() && group.size(0) <= reference.size(0)) {
        std::vector<int64_t> pad_shape(reference.sizes().vec());
        pad_shape[0] = reference.size(0) - group.size(0);
        auto padded = torch::zeros(pad_shape, reference.options());
        group = torch::cat({group.detach(), padded}, 0);
        return;
      }
      group = torch::zeros_like(reference);
    };
    auto accumulate_group = [](torch::Tensor& dst, const torch::Tensor& src) {
      if (!src.defined() || src.numel() == 0) {
        return;
      }
      if (!dst.defined() || dst.numel() == 0) {
        dst = src.detach().clone();
      } else {
        dst.add_(src.detach());
      }
    };
    auto scale_group = [](const torch::Tensor& group, const float scale) {
      if (!group.defined() || group.numel() == 0) {
        return torch::Tensor();
      }
      return group * scale;
    };
    auto blend_group = [&](torch::Tensor& dst, const torch::Tensor& src) {
      if (!src.defined() || src.numel() == 0) {
        return;
      }
      if (!dst.defined() || dst.numel() == 0) {
        dst = src.detach().clone();
      } else {
        dst.mul_(1.0f - ema_alpha).add_(src.detach(), ema_alpha);
      }
    };
    auto make_diag = [](const gs::FisherTrainingDiag& diag) {
      GsFisherDiag out;
      out.xyz = diag.xyz;
      out.opacity = diag.opacity;
      return out;
    };
    auto pad_diag = [&](GsFisherDiag& diag) {
      pad_group(diag.xyz, target_xyz);
      pad_group(diag.opacity, target_opacity);
    };

    pad_group(gs_state.fisherHistDiag.xyz, target_xyz);
    pad_group(gs_state.fisherHistDiag.opacity, target_opacity);

    GsFisherDiag sampled_hist_sum;
    size_t sampled_hist_count = 0;
    for (const auto& sample : fisher_training_samples) {
      GsFisherDiag sample_diag = make_diag(sample.diag);
      pad_diag(sample_diag);
      if ((!sample_diag.xyz.defined() || sample_diag.xyz.numel() == 0)
          && (!sample_diag.opacity.defined() || sample_diag.opacity.numel() == 0)) {
        continue;
      }

      if (sample.is_current_frame && sample.contributes_to_history) {
        gs_state.fisherHistDiag = addGsFisherDiag(gs_state.fisherHistDiag, sample_diag);
        continue;
      }

      if (sample.contributes_to_history) {
        accumulate_group(sampled_hist_sum.xyz, sample_diag.xyz);
        accumulate_group(sampled_hist_sum.opacity, sample_diag.opacity);
        sampled_hist_count++;
      }
    }

    const size_t history_view_count =
        std::min(gs_state.gsCamListCpu.size(), gs_state.gtImgListCpu.size());
    if (sampled_hist_count > 0 && history_view_count > 0) {
      const float scale =
          static_cast<float>(history_view_count) / static_cast<float>(sampled_hist_count);
      blend_group(gs_state.fisherHistDiag.xyz, scale_group(sampled_hist_sum.xyz, scale));
      blend_group(gs_state.fisherHistDiag.opacity,
                  scale_group(sampled_hist_sum.opacity, scale));
      gs_state.fisherEstimateUpdateCount++;
    }

    gs_state.fisherHistViewCount = history_view_count;
    gs_state.fisherCacheVersion++;
  }

  torch::Tensor SubmappingInterface::collapseGsFisherDiagPerGaussian(
      const GsFisherDiag& fisher_diag, const GsFisherScoreConfig& score_config) const
  {
    torch::Tensor per_gaussian;
    auto accumulate_group =
        [&](const torch::Tensor& fisher_group, const float weight) {
          if (!fisher_group.defined() || fisher_group.numel() == 0) {
            return;
          }
          auto group_mean =
              fisher_group.detach().reshape({fisher_group.size(0), -1}).mean(1) * weight;
          if (!per_gaussian.defined()) {
            per_gaussian = group_mean.clone();
          } else {
            per_gaussian.add_(group_mean);
          }
        };

    accumulate_group(fisher_diag.xyz, score_config.weight_xyz);
    accumulate_group(fisher_diag.opacity, score_config.weight_opacity);
    return per_gaussian;
  }

  torch::Tensor SubmappingInterface::normalizePerGaussianScoreForDisplay(
      const torch::Tensor& per_gaussian_score) const
  {
    if (!per_gaussian_score.defined() || per_gaussian_score.numel() == 0) {
      return torch::Tensor();
    }

    auto score_log =
        torch::log1p(torch::clamp_min(per_gaussian_score.detach(), 0.0f)).to(torch::kCPU).contiguous();

    std::vector<float> score_values;
    score_values.reserve(static_cast<size_t>(score_log.numel()));
    const auto* score_values_ptr = score_log.data_ptr<float>();
    for (int64_t i = 0; i < score_log.numel(); ++i) {
      const float value = score_values_ptr[i];
      if (std::isfinite(value)) {
        score_values.push_back(value);
      }
    }

    if (score_values.empty()) {
      return torch::zeros_like(score_log);
    }

    std::sort(score_values.begin(), score_values.end());
    const size_t low_idx = static_cast<size_t>(
        std::floor(0.05 * static_cast<double>(score_values.size() - 1)));
    const size_t high_idx = static_cast<size_t>(
        std::floor(0.98 * static_cast<double>(score_values.size() - 1)));
    const float low_value = score_values[low_idx];
    const float high_value = score_values[std::max(low_idx, high_idx)];

    if (high_value > low_value + 1.0e-6f) {
      return ((score_log - low_value) / (high_value - low_value)).clamp(0.0f, 1.0f);
    }

    const float max_value = score_values.back();
    if (max_value > 1.0e-6f) {
      return (score_log / max_value).clamp(0.0f, 1.0f);
    }

    return torch::zeros_like(score_log);
  }

  void SubmappingInterface::clearInactiveFisherOverlays(const uint64_t active_submap_id)
  {
    for (auto& cache_kv : occupancySubmapCaches_) {
      if (cache_kv.first == active_submap_id) {
        continue;
      }
      auto& cache = cache_kv.second;
      const bool had_fisher_colours =
          static_cast<bool>(cache.occupied_fisher_colours);
      const bool had_fisher_grid = !cache.fisher_score_grid.empty();
      if (!had_fisher_colours && !had_fisher_grid && cache.fisher_cell_size <= 1.0e-6f) {
        continue;
      }
      cache.fisher_score_grid.clear();
      cache.fisher_cell_size = 0.0f;
      cache.occupied_fisher_colours.reset();
      cache.colour_revision++;
    }
  }

  void SubmappingInterface::clearGlobalFisherOverlay()
  {
    globalFisherScoreGrid_.clear();
    globalFisherDisplayGrid_.clear();
    globalFisherCellSize_ = 0.0f;

    for (auto& cache_kv : occupancySubmapCaches_) {
      auto& cache = cache_kv.second;
      const bool had_global_fisher_colours =
          static_cast<bool>(cache.occupied_global_fisher_colours);
      const bool had_global_fisher_grid = !cache.global_fisher_score_grid.empty();
      if (!had_global_fisher_colours && !had_global_fisher_grid
          && cache.global_fisher_cell_size <= 1.0e-6f) {
        continue;
      }
      cache.global_fisher_score_grid.clear();
      cache.global_fisher_cell_size = 0.0f;
      cache.occupied_global_fisher_colours.reset();
      cache.colour_revision++;
    }
  }

  void SubmappingInterface::normalizeGlobalFisherScoreGridForDisplay()
  {
    globalFisherDisplayGrid_.clear();
    if (globalFisherScoreGrid_.empty()) {
      return;
    }

    std::vector<float> score_values;
    score_values.reserve(globalFisherScoreGrid_.size());
    for (const auto& score_kv : globalFisherScoreGrid_) {
      const float value = std::log1p(std::max(score_kv.second, 0.0f));
      if (std::isfinite(value)) {
        score_values.push_back(value);
      }
    }

    if (score_values.empty()) {
      return;
    }

    std::sort(score_values.begin(), score_values.end());
    const size_t low_idx = static_cast<size_t>(
        std::floor(0.05 * static_cast<double>(score_values.size() - 1)));
    const size_t high_idx = static_cast<size_t>(
        std::floor(0.98 * static_cast<double>(score_values.size() - 1)));
    const float low_value = score_values[low_idx];
    const float high_value = score_values[std::max(low_idx, high_idx)];
    const float max_value = score_values.back();

    globalFisherDisplayGrid_.reserve(globalFisherScoreGrid_.size());
    for (const auto& score_kv : globalFisherScoreGrid_) {
      const float value = std::log1p(std::max(score_kv.second, 0.0f));
      float normalized = 0.0f;
      if (high_value > low_value + 1.0e-6f) {
        normalized = clampUnit((value - low_value) / (high_value - low_value));
      } else if (max_value > 1.0e-6f) {
        normalized = clampUnit(value / max_value);
      }
      globalFisherDisplayGrid_.emplace(score_kv.first, normalized);
    }
  }

  void SubmappingInterface::rebuildGlobalFisherOverlay()
  {
    globalFisherScoreGrid_.clear();
    globalFisherDisplayGrid_.clear();
    globalFisherCellSize_ = 0.0f;

    for (const auto& cache_kv : occupancySubmapCaches_) {
      const auto& cache = cache_kv.second;
      if (cache.global_fisher_score_grid.empty() || cache.global_fisher_cell_size <= 1.0e-6f) {
        continue;
      }
      if (globalFisherCellSize_ <= 1.0e-6f) {
        globalFisherCellSize_ = cache.global_fisher_cell_size;
      }
      for (const auto& fisher_kv : cache.global_fisher_score_grid) {
        auto& score = globalFisherScoreGrid_[fisher_kv.first];
        score = std::max(score, fisher_kv.second);
      }
    }

    normalizeGlobalFisherScoreGridForDisplay();

    for (auto& cache_kv : occupancySubmapCaches_) {
      recolorOccupancySubmapFromGlobalFisher(cache_kv.first, cache_kv.second);
      cache_kv.second.colour_revision++;
    }
  }

  bool SubmappingInterface::projectAllGsFisherToGlobalOccupancy()
  {
    if (!gsRuntime_) {
      return false;
    }

    const float fisher_eig_regularizer =
        std::max(1.0e-12f, occupancyViewerConfig_.fisher_eig_regularizer);
    std::unordered_set<uint64_t> refreshed_submaps;
    refreshed_submaps.reserve(seSubmapLookup_.size());
    bool touched_any_cache = false;

    for (const auto& submap_kv : seSubmapLookup_) {
      const uint64_t submap_id = submap_kv.first;
      const auto& submap = submap_kv.second;
      if (!submap.map) {
        continue;
      }

      const auto state_it = gsRuntime_->submapStates.find(submap_id);
      if (state_it == gsRuntime_->submapStates.end()) {
        continue;
      }

      auto& gs_state = state_it->second;
      torch::Tensor xyz_local_cpu;
      try {
        xyz_local_cpu = gsSubmapXyzCpu(submap_id);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Skipping global GS Fisher projection for submap " << submap_id
                     << " while extracting xyz tensor: " << e.what();
        continue;
      } catch (...) {
        LOG(WARNING) << "Skipping global GS Fisher projection for submap " << submap_id
                     << " while extracting xyz tensor due to an unknown exception.";
        continue;
      }

      if (!gsTensorDefinedNonEmpty(xyz_local_cpu)) {
        continue;
      }

      auto& cache = occupancySubmapCaches_[submap_id];
      refreshed_submaps.insert(submap_id);
      touched_any_cache = true;

      const GsFisherDiag prior_inv =
          regularizeInvertGsFisherDiag(gs_state.fisherHistDiag, fisher_eig_regularizer);
      torch::Tensor raw_score_per_gaussian =
          collapseGsFisherDiagPerGaussian(prior_inv, gsFisherScoreConfig_);
      if (!raw_score_per_gaussian.defined()
          || raw_score_per_gaussian.numel() != xyz_local_cpu.size(0)) {
        cache.global_fisher_score_grid.clear();
        cache.global_fisher_cell_size = 0.0f;
        continue;
      }

      auto raw_values_cpu =
          raw_score_per_gaussian.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
      const auto* xyz_ptr = xyz_local_cpu.data_ptr<float>();
      const auto* raw_ptr = raw_values_cpu.data_ptr<float>();
      const float cell_size =
          std::max(submap.map->getRes() * static_cast<float>(occupancyViewerConfig_.voxel_stride),
                   1.0e-3f);
      const Eigen::Matrix4f& T_WK = submap.T_WK.matrix();
      const auto makeKey = [&](const Eigen::Vector3f& p_world) {
        return OccupancyBlockKey{
            static_cast<int>(std::floor(p_world.x() / cell_size)),
            static_cast<int>(std::floor(p_world.y() / cell_size)),
            static_cast<int>(std::floor(p_world.z() / cell_size))};
      };

      std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash> global_fisher_grid;
      global_fisher_grid.reserve(static_cast<size_t>(xyz_local_cpu.size(0)));
      for (int64_t i = 0; i < xyz_local_cpu.size(0); ++i) {
        const Eigen::Vector3f p_local(xyz_ptr[3 * i + 0], xyz_ptr[3 * i + 1], xyz_ptr[3 * i + 2]);
        const Eigen::Vector3f p_world = (T_WK * p_local.homogeneous()).head<3>();
        const OccupancyBlockKey global_key = makeKey(p_world);
        auto& global_cell_score = global_fisher_grid[global_key];
        global_cell_score = std::max(global_cell_score, std::max(raw_ptr[i], 0.0f));
      }

      cache.global_fisher_score_grid = std::move(global_fisher_grid);
      cache.global_fisher_cell_size = cell_size;
    }

    for (auto& cache_kv : occupancySubmapCaches_) {
      if (refreshed_submaps.count(cache_kv.first) > 0) {
        continue;
      }
      auto& cache = cache_kv.second;
      if (cache.global_fisher_score_grid.empty() && cache.global_fisher_cell_size <= 1.0e-6f) {
        continue;
      }
      touched_any_cache = true;
      cache.global_fisher_score_grid.clear();
      cache.global_fisher_cell_size = 0.0f;
    }

    rebuildGlobalFisherOverlay();
    return touched_any_cache;
  }

  bool SubmappingInterface::updateGsActiveFisherVisualization()
  {
    if (!gsFisherVisualizationEnabled_.load()) {
      if (gsFisherVisualizationRefreshRequested_.exchange(false)) {
        clearInactiveFisherOverlays(UNINITIALIZED_ID);
        return true;
      }
      return false;
    }

    if (!gsRuntime_ || !gsMappingEnabled_.load() || prevKeyframeId_ == UNINITIALIZED_ID) {
      return false;
    }

    const uint64_t update_interval =
        static_cast<uint64_t>(std::max(1, occupancyViewerConfig_.fisher_update_interval));
    const bool force_update = gsFisherVisualizationRefreshRequested_.exchange(false);
    if (!force_update
        && (integration_counter_ % update_interval != 0
            || gsFisherLastUpdateCounter_ == integration_counter_)) {
      return false;
    }

    gsFisherLastUpdateCounter_ = integration_counter_;
    clearInactiveFisherOverlays(prevKeyframeId_);
    try {
      const auto submap_it = seSubmapLookup_.find(prevKeyframeId_);
      const auto state_it = gsRuntime_->submapStates.find(prevKeyframeId_);
      auto cache_it = occupancySubmapCaches_.find(prevKeyframeId_);
      if (submap_it == seSubmapLookup_.end() || !submap_it->second.map
          || state_it == gsRuntime_->submapStates.end()
          || cache_it == occupancySubmapCaches_.end()
          || !cache_it->second.occupied_points_local
          || cache_it->second.occupied_points_local->empty()) {
        clearInactiveFisherOverlays(UNINITIALIZED_ID);
        return true;
      }

      const torch::Tensor xyz_local_cpu = gsSubmapXyzCpu(prevKeyframeId_);
      const GsFisherDiag prior_inv = regularizeInvertGsFisherDiag(
          state_it->second.fisherHistDiag, occupancyViewerConfig_.fisher_eig_regularizer);
      const torch::Tensor prior_score_per_gaussian =
          collapseGsFisherDiagPerGaussian(prior_inv, gsFisherScoreConfig_);
      if (!gsTensorDefinedNonEmpty(xyz_local_cpu)
          || !gsTensorDefinedNonEmpty(prior_score_per_gaussian)
          || prior_score_per_gaussian.numel() != xyz_local_cpu.size(0)) {
        clearInactiveFisherOverlays(UNINITIALIZED_ID);
        return true;
      }
      const torch::Tensor display_values_cpu =
          normalizePerGaussianScoreForDisplay(prior_score_per_gaussian);
      const auto* xyz_ptr = xyz_local_cpu.data_ptr<float>();
      const auto* display_ptr = display_values_cpu.data_ptr<float>();
      const float cell_size = std::max(
          submap_it->second.map->getRes() * static_cast<float>(occupancyViewerConfig_.voxel_stride),
          1.0e-3f);

      std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash> fisher_grid;
      fisher_grid.reserve(static_cast<size_t>(xyz_local_cpu.size(0)));
      for (int64_t i = 0; i < xyz_local_cpu.size(0); ++i) {
        const OccupancyBlockKey key{
            static_cast<int>(std::floor(xyz_ptr[3 * i] / cell_size)),
            static_cast<int>(std::floor(xyz_ptr[3 * i + 1] / cell_size)),
            static_cast<int>(std::floor(xyz_ptr[3 * i + 2] / cell_size))};
        auto& cell_score = fisher_grid[key];
        cell_score = std::max(cell_score, display_ptr[i]);
      }

      auto& cache = cache_it->second;
      cache.fisher_score_grid = std::move(fisher_grid);
      cache.fisher_cell_size = cell_size;
      recolorOccupancySubmapFromFisher(cache);
      cache.colour_revision++;
      return true;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Active submap Fisher visualization update failed: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Active submap Fisher visualization update failed with unknown exception.";
    }
    clearInactiveFisherOverlays(UNINITIALIZED_ID);
    return true;
  }

  bool SubmappingInterface::updateGsGlobalFisherVisualization()
  {

    if (!gsGlobalFisherVisualizationEnabled_.load()) {
      if (gsGlobalFisherVisualizationRefreshRequested_.exchange(false)) {
        clearGlobalFisherOverlay();
        return true;
      }
      return false;
    }

    if (!gsRuntime_ || !gsMappingEnabled_.load() || seSubmapLookup_.empty()) {
      return false;
    }

    const uint64_t update_interval =
        static_cast<uint64_t>(std::max(1, occupancyViewerConfig_.fisher_update_interval));
    const bool force_update = gsGlobalFisherVisualizationRefreshRequested_.exchange(false);
    if (!force_update
        && (integration_counter_ % update_interval != 0
            || gsGlobalFisherLastUpdateCounter_ == integration_counter_)) {
      return false;
    }

    try {
      const bool updated = projectAllGsFisherToGlobalOccupancy();
      gsGlobalFisherLastUpdateCounter_ = integration_counter_;
      return updated;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Global Fisher visualization update failed: " << e.what();
      clearGlobalFisherOverlay();
      gsGlobalFisherLastUpdateCounter_ = integration_counter_;
      return true;
    } catch (...) {
      LOG(WARNING) << "Global Fisher visualization update failed with unknown exception.";
      clearGlobalFisherOverlay();
      gsGlobalFisherLastUpdateCounter_ = integration_counter_;
      return true;
    }
  }

  void SubmappingInterface::updateGsFisherVisualization()
  {
    const bool active_updated = updateGsActiveFisherVisualization();
    const bool global_updated = updateGsGlobalFisherVisualization();
    if ((active_updated || global_updated) && occupancyViewerConfig_.enable) {
      OccupancyPreviewFrame fisher_preview_frame;
      if (buildOccupancyPreviewSnapshot(fisher_preview_frame, false)) {
        occupancyPreviewQueue_.PushNonBlockingDroppingIfFull(
            fisher_preview_frame, kOccupancyPreviewQueueSize);
      }
    }
  }

  void SubmappingInterface::updateOccupancyPreview(const State& current_state)
  {
    if (seSubmapLookup_.empty()) {
      return;
    }
    const bool publish_preview = occupancyViewerConfig_.enable;
    const int update_interval =
        publish_preview ? std::max(1, occupancyViewerConfig_.update_interval) : 1;
    if (integration_counter_ % static_cast<uint64_t>(update_interval) != 0) {
      return;
    }

    constexpr size_t kMaxDirtyBlocksPerSubmapPerUpdate = 24;
    const int max_points = publish_preview ? occupancyViewerConfig_.max_points : 0;
    const int stride = publish_preview ? occupancyViewerConfig_.voxel_stride : 1;
    const int num_submaps = std::max(1, static_cast<int>(seSubmapLookup_.size()));
    const int per_submap_occ_budget = std::max(1, max_points / num_submaps);

    for (auto it = occupancySubmapCaches_.begin(); it != occupancySubmapCaches_.end();) {
      if (seSubmapLookup_.find(it->first) == seSubmapLookup_.end()) {
        occupancyDirtySubmaps_.erase(it->first);
        it = occupancySubmapCaches_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = occupancyDirtySubmaps_.begin(); it != occupancyDirtySubmaps_.end();) {
      if (seSubmapLookup_.find(*it) == seSubmapLookup_.end()) {
        it = occupancyDirtySubmaps_.erase(it);
      } else {
        ++it;
      }
    }

    std::unordered_set<uint64_t> geometry_updated_submaps;
    geometry_updated_submaps.reserve(seSubmapLookup_.size());
    for (const auto& submap_kv : seSubmapLookup_) {
      const uint64_t submap_id = submap_kv.first;
      const auto& submap = submap_kv.second;
      if (!submap.map) {
        continue;
      }

      auto cache_it = occupancySubmapCaches_.find(submap_id);
      const bool cache_missing = (cache_it == occupancySubmapCaches_.end());
      if (cache_missing) {
        cache_it = occupancySubmapCaches_.emplace(submap_id, OccupancySubmapCache{}).first;
      }
      auto& cache = cache_it->second;
      const bool cache_mismatch =
          (cache.cached_stride != stride || cache.cached_occ_budget != per_submap_occ_budget);

      if (cache_missing || cache_mismatch || cache.needs_full_rebuild) {
        if (rebuildOccupancySubmapCache(submap_id,
                                        submap,
                                        stride,
                                        per_submap_occ_budget)) {
          geometry_updated_submaps.insert(submap_id);
          occupancyDirtySubmaps_.erase(submap_id);
        }
        continue;
      }

      if (!occupancyDirtySubmaps_.count(submap_id)) {
        continue;
      }

      bool cache_changed = false;
      size_t processed_blocks = 0;
      while (processed_blocks < kMaxDirtyBlocksPerSubmapPerUpdate
             && !cache.dirty_block_queue.empty()) {
        const OccupancyBlockKey block_key = cache.dirty_block_queue.front();
        cache.dirty_block_queue.pop_front();
        cache.dirty_block_set.erase(block_key);

        OccupancyTrackedBlock sampled_block;
        const bool block_exists =
            sampleOccupancyBlock(submap, block_key, stride, sampled_block);
        const auto tracked_it = cache.tracked_block_indices.find(block_key);
        if (block_exists) {
          if (tracked_it != cache.tracked_block_indices.end()) {
            cache.tracked_blocks[tracked_it->second] = std::move(sampled_block);
            cache.tracked_blocks[tracked_it->second].active = true;
            cache_changed = true;
          } else {
            cache.tracked_block_indices.emplace(block_key, cache.tracked_blocks.size());
            cache.tracked_blocks.push_back(std::move(sampled_block));
            cache_changed = true;
          }
        } else if (tracked_it != cache.tracked_block_indices.end()) {
          cache.tracked_blocks[tracked_it->second].active = false;
          cache.tracked_block_indices.erase(tracked_it);
          cache_changed = true;
        }
        processed_blocks++;
      }

      if (cache_changed) {
        if (!cache.tracked_blocks.empty() && cache.tracked_blocks.size() > 1024) {
          std::vector<OccupancyTrackedBlock> compacted_blocks;
          compacted_blocks.reserve(cache.tracked_block_indices.size());
          cache.tracked_block_indices.clear();
          for (const auto& tracked_block : cache.tracked_blocks) {
            if (!tracked_block.active) {
              continue;
            }
            cache.tracked_block_indices.emplace(tracked_block.key, compacted_blocks.size());
            compacted_blocks.push_back(tracked_block);
          }
          cache.tracked_blocks.swap(compacted_blocks);
        }
        rebuildOccupancySubmapPoints(
            submap_id, cache, per_submap_occ_budget);
        cache.geometry_revision++;
        geometry_updated_submaps.insert(submap_id);
      }

      if (cache.dirty_block_queue.empty()) {
        occupancyDirtySubmaps_.erase(submap_id);
      }
    }

    if (!publish_preview) {
      return;
    }

    OccupancyPreviewFrame preview_frame;
    auto submap_clouds =
        std::make_shared<std::vector<OccupancyPreviewFrame::SubmapCloud>>();
    submap_clouds->reserve(seSubmapLookup_.size());
    size_t total_occ_points = 0;
    for (const auto& submap_kv : seSubmapLookup_) {
      const uint64_t submap_id = submap_kv.first;
      const auto& submap = submap_kv.second;
      if (!submap.map) {
        continue;
      }
      auto cache_it = occupancySubmapCaches_.find(submap_id);
      if (cache_it == occupancySubmapCaches_.end()) {
        continue;
      }
      OccupancyPreviewFrame::SubmapCloud submap_cloud;
      submap_cloud.submap_id = submap_id;
      submap_cloud.occupied_points_local = cache_it->second.occupied_points_local;
      submap_cloud.occupied_colours = cache_it->second.occupied_colours;
      submap_cloud.occupied_fisher_colours = cache_it->second.occupied_fisher_colours;
      submap_cloud.occupied_global_fisher_colours =
          cache_it->second.occupied_global_fisher_colours;
      submap_cloud.T_WK = submap.T_WK.matrix();
      submap_cloud.num_points = cache_it->second.num_points;
      submap_cloud.geometry_revision = cache_it->second.geometry_revision;
      submap_cloud.colour_revision = cache_it->second.colour_revision;
      submap_cloud.geometry_updated = geometry_updated_submaps.count(submap_id) > 0;
      total_occ_points += submap_cloud.num_points;
      submap_clouds->push_back(std::move(submap_cloud));
    }
    preview_frame.submap_clouds = submap_clouds;
    preview_frame.T_WS = current_state.T_WS.T().cast<float>();
    preview_frame.num_submaps = seSubmapLookup_.size();
    preview_frame.num_points = total_occ_points;

    preview_frame.integration_counter = integration_counter_;
    occupancyPreviewQueue_.PushNonBlockingDroppingIfFull(preview_frame,
                                                          kOccupancyPreviewQueueSize);
  }

  void SubmappingInterface::enqueueSubmapFinalization(const uint64_t submap_id)
  {
    if (submap_id == UNINITIALIZED_ID) {
      return;
    }
    const auto submap_it = seSubmapLookup_.find(submap_id);
    if (submap_it == seSubmapLookup_.end() || !submap_it->second.map) {
      return;
    }

    SubmapFinalizeTask task;
    task.submap_id = submap_id;
    task.map = submap_it->second.map;

    {
      std::lock_guard<std::mutex> lock(submapFinalizeMutex_);
      if (submapFinalizeStop_ || submapFinalizePending_.count(submap_id)) {
        return;
      }
      submapFinalizeQueue_.push_back(std::move(task));
      submapFinalizePending_.insert(submap_id);
    }
    submapFinalizeCv_.notify_one();
  }


  void SubmappingInterface::runSubmapFinalizationWorker()
  {
    while (true) {
      SubmapFinalizeTask task;
      {
        std::unique_lock<std::mutex> lock(submapFinalizeMutex_);
        submapFinalizeCv_.wait(lock, [this]() {
          return submapFinalizeStop_ || !submapFinalizeQueue_.empty();
        });
        if (submapFinalizeStop_ && submapFinalizeQueue_.empty()) {
          return;
        }
        task = std::move(submapFinalizeQueue_.front());
        submapFinalizeQueue_.pop_front();
        submapFinalizeInFlight_++;
      }

      try {
        if (task.map) {
          auto mesh = task.map->mesh();
          std::lock_guard<std::mutex> mesh_lock(submapMeshMutex_);
          seMeshLookup_[task.submap_id] = std::move(mesh);
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Asynchronous submap finalization failed for submap "
                   << task.submap_id << ": " << e.what();

      } catch (...) {
        LOG(ERROR) << "Asynchronous submap finalization failed for submap "
                   << task.submap_id << " with unknown exception.";

      }

      {
        std::lock_guard<std::mutex> lock(submapFinalizeMutex_);
        if (submapFinalizeInFlight_ > 0) {
          submapFinalizeInFlight_--;
        }
        submapFinalizePending_.erase(task.submap_id);
        if (submapFinalizeQueue_.empty() && submapFinalizeInFlight_ == 0) {
          submapFinalizeCv_.notify_all();
        }
      }
    }
  }

  void SubmappingInterface::waitForSubmapFinalization()
  {
    if (!submapFinalizeWorker_.joinable()) {
      std::deque<SubmapFinalizeTask> queued_tasks;
      {
        std::lock_guard<std::mutex> lock(submapFinalizeMutex_);
        queued_tasks.swap(submapFinalizeQueue_);
        submapFinalizePending_.clear();
        submapFinalizeInFlight_ = 0;
      }
      for (auto& task : queued_tasks) {
        if (!task.map) {
          continue;
        }
        try {
          auto mesh = task.map->mesh();

          std::lock_guard<std::mutex> object_lock(submapMeshMutex_);
          seMeshLookup_[task.submap_id] = std::move(mesh);

        } catch (const std::exception& e) {
          LOG(ERROR) << "Synchronous submap finalization fallback failed for submap "
                     << task.submap_id << ": " << e.what();
        }
      }
      return;
    }

    std::unique_lock<std::mutex> lock(submapFinalizeMutex_);
    submapFinalizeCv_.wait(lock, [this]() {
      return submapFinalizeQueue_.empty() && submapFinalizeInFlight_ == 0;
    });
  }


  bool SubmappingInterface::clearFrozenGsTrainingResources(const uint64_t submap_id)
  {
    if (!gsRuntime_) {
      return false;
    }
    const auto state_it = gsRuntime_->submapStates.find(submap_id);
    if (state_it == gsRuntime_->submapStates.end()) {
      return false;
    }

    auto& gs_state = state_it->second;
    bool released_gpu_memory = false;
    if (!gs_state.gsCamList.empty()) {
      std::vector<gs::Camera>().swap(gs_state.gsCamList);
      released_gpu_memory = true;
    }
    if (!gs_state.gtImgList.empty()) {
      std::vector<torch::Tensor>().swap(gs_state.gtImgList);
      released_gpu_memory = true;
    }
    if (!gs_state.geometryTargetList.empty()) {
      std::vector<gs::GeometryTarget>().swap(gs_state.geometryTargetList);
      released_gpu_memory = true;
    }
    if (gs_state.model && gs_state.model->optimizer) {
      gs_state.model->Release_training_state(true);
      released_gpu_memory = true;
    }
    return released_gpu_memory;
  }

  bool SubmappingInterface::enqueueGsModelOffload(const uint64_t submap_id)
  {
    if (!gsRuntime_) {
      return false;
    }
    GsModelOffloadTask task;
    {
      std::lock_guard<std::mutex> lock(gsModelOffloadMutex_);
      if (gsModelOffloadStop_) {
        return false;
      }
      const auto state_it = gsRuntime_->submapStates.find(submap_id);
      if (state_it == gsRuntime_->submapStates.end()) {
        return false;
      }

      auto& gs_state = state_it->second;
      if (gs_state.offloadQueued || !gs_state.model) {
        return false;
      }

      task.submap_id = submap_id;
      task.runtime = gsRuntime_;
      task.model = std::move(gs_state.model);
      gs_state.offloadQueued = true;
      gsModelOffloadQueue_.push_back(std::move(task));
    }
    return true;
  }


  void SubmappingInterface::waitForGsModelOffloads()
  {
    std::deque<GsModelOffloadTask> queued_tasks;
    {
      std::lock_guard<std::mutex> lock(gsModelOffloadMutex_);
      queued_tasks.swap(gsModelOffloadQueue_);
    }
    for (auto& task : queued_tasks) {
      if (!task.runtime || !task.model) continue;
      auto* gs_state = &task.runtime->submapStates.at(task.submap_id);
      try {
        gs_state->cpuModelSnapshot = task.model->Export_ply_tensors_cpu();
        gs_state->hasCpuModelSnapshot = true;
      } catch (...) {
        gs_state->model = std::move(task.model);
      }
      gs_state->offloadQueued = false;
    }
  }

  void SubmappingInterface::trimGsCudaCache()
  {
    torch::cuda::synchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  void SubmappingInterface::updateGsSubmapResidencyPolicy(const uint64_t active_submap_id)
  {
    if (!gsRuntime_ || !submapConfig_.gsOptimizeCudaMemory) {
      return;
    }

    const int keep_recent =
        std::max(0, submapConfig_.gsKeepRecentFrozenSubmapsGpu);
    std::vector<std::pair<size_t, uint64_t>> frozen_submaps;
    frozen_submaps.reserve(gsRuntime_->submapStates.size());
    for (const auto& kv : gsRuntime_->submapStates) {
      if (kv.first == active_submap_id) {
        continue;
      }
      frozen_submaps.emplace_back(kv.second.creationIndex, kv.first);
    }
    std::sort(frozen_submaps.begin(),
              frozen_submaps.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (const auto& entry : frozen_submaps) {
      clearFrozenGsTrainingResources(entry.second);
    }

    const int offload_count =
        std::max(0, static_cast<int>(frozen_submaps.size()) - keep_recent);
    for (int i = 0; i < offload_count; ++i) {
      enqueueGsModelOffload(frozen_submaps[i].second);
    }
    waitForGsModelOffloads();
  }

  bool SubmappingInterface::finalizeGsMapping(
      const std::string& datasetPath,
      const std::string& finalTrajectoryCsv,
      const std::vector<ViSlamBackend::FinalTrajectoryPosition>& finalTrajectoryPoses)
  {
    if (!gsRuntime_) {
      if (!gsMappingEnabled_.load()) {
        return true;
      }
      LOG(WARNING) << "GS finalization skipped: runtime not initialized.";
      return false;
    }

    waitForGsModelOffloads();

    {
      std::lock_guard<std::mutex> lock(gsFinalizationSummaryMtx_);
      lastGsFinalizationSummary_ = GsFinalizationSummary();
    }

    try {
      PoseLookup final_poses_by_timestamp;
      PoseLookup final_poses_by_state_id;
      std::vector<std::pair<uint64_t, std::string>> rgb_frames;
      double pre_global_opt_system_fps = -1.0;
      size_t global_opt_kf_count = 0;
      double global_optimization_sec_summary = 0.0;
      bool fisher_export_ok = true;
      size_t fisher_submaps_available = 0;
      size_t fisher_submaps_unavailable = 0;
      const bool render_requested = submapConfig_.gsSaveRenderedRgb;
      const std::filesystem::path render_output_directory =
          submapConfig_.gsRenderedRgbDirectory.empty()
              ? (gsRuntime_->outputDirectory / "rendered_rgb")
              : std::filesystem::path(submapConfig_.gsRenderedRgbDirectory);
      size_t render_saved_count = 0;
      size_t render_pose_miss_count = 0;
      bool joint_pose_applied = false;
      size_t joint_pose_updated_rows = 0;
      const bool need_final_fisher_snapshot =
          submapConfig_.gsSaveFisher || occupancyViewerConfig_.enable;

      bool have_final_poses =
          loadFinalTrajectoryPosesFromProvider(finalTrajectoryPoses,
                                               final_poses_by_timestamp,
                                               final_poses_by_state_id);
      if (!have_final_poses) {
        have_final_poses = loadFinalTrajectoryPoses(finalTrajectoryCsv, final_poses_by_timestamp);
        if (have_final_poses) {
          LOG(INFO) << "GS finalization: using CSV pose fallback from " << finalTrajectoryCsv;
        }
      } else {
        LOG(INFO) << "GS finalization: using in-memory final pose provider with "
                  << final_poses_by_timestamp.size() << " poses.";
      }

      gsOptimizedTrajectoryCallback gs_optimized_traj_callback;
      {
        std::lock_guard<std::mutex> lock(gsOptimizedTrajectoryCallbackMtx_);
        gs_optimized_traj_callback = gsOptimizedTrajectoryCallback_;
      }
      auto publishGsOptimizedTrajectory = [&](const PoseLookup& poses_by_state_id) {
        if (!gs_optimized_traj_callback || poses_by_state_id.empty()) {
          return;
        }
        std::map<uint64_t, Eigen::Vector3f> trajectory_by_state_id;
        for (const auto& pose_it : poses_by_state_id) {
          trajectory_by_state_id[pose_it.first] = pose_it.second.translation();
        }
        gs_optimized_traj_callback(trajectory_by_state_id);
      };
      publishGsOptimizedTrajectory(final_poses_by_state_id);

      bool have_rgb_frames = false;

      const std::string rgb_stream =
          viParameters_.mappingInputPreprocessor
              ? viParameters_.mappingInputPreprocessor->rgbStream()
              : std::string("rgb0");
      {
        const std::filesystem::path rgb_index_path =
            std::filesystem::path(datasetPath) / rgb_stream / "data.csv";
        if (std::filesystem::exists(rgb_index_path)) {
          have_rgb_frames = loadRgbDatasetIndex(
              datasetPath,
              rgb_stream,
              viParameters_.camera.image_delay,
              rgb_frames);
        }
      }
      if (have_rgb_frames && viParameters_.camera.image_delay != 0.0) {
        LOG(INFO) << "GS finalization: applied image_delay="
                  << viParameters_.camera.image_delay
                  << " s to final-render timestamps for final-pose matching.";
      }

      size_t final_iteration_raw = integration_counter_;
      if (final_iteration_raw == 0) {
        LOG(WARNING) << "GS finalization: integration_counter is 0. "
                        "Falling back to iteration_1 for export naming.";
        final_iteration_raw = 1;
      }
      const int finalIteration = static_cast<int>(final_iteration_raw);
      LOG(INFO) << "GS finalization iterations: integration_counter=" << integration_counter_
                << " trajectory_poses=" << final_poses_by_timestamp.size()
                << " rgb_frames=" << rgb_frames.size()
                << " -> save_iteration=" << finalIteration;

      bool has_system_start = false;
      double system_elapsed_sec = 0.0;
      {
        std::lock_guard<std::mutex> lock(mappingFpsMtx_);
        has_system_start = mappingFpsTrackingStartSet_;
        if (has_system_start) {
          system_elapsed_sec = (okvis::Time::now() - mappingFpsTrackingStartTime_).toSec();
        }
      }
      if (has_system_start && system_elapsed_sec > 1.0e-6 && integration_counter_ > 0) {
        const double system_fps = static_cast<double>(integration_counter_) / system_elapsed_sec;
        pre_global_opt_system_fps = system_fps;
        LOG(INFO) << std::fixed << std::setprecision(3)
                  << "GS pre-global-opt average System FPS: " << system_fps
                  << " (frames=" << integration_counter_
                  << ", elapsed=" << system_elapsed_sec << " s)";
      } else {
        LOG(WARNING) << "GS pre-global-opt average System FPS unavailable "
                     << "(start_set=" << has_system_start
                     << ", frames=" << integration_counter_
                     << ", elapsed=" << system_elapsed_sec << " s)";
      }

      size_t gs_pose_samples_total = 0;
      size_t gs_pose_samples_matched = 0;
      size_t gs_pose_samples_matched_state = 0;
      size_t gs_pose_samples_matched_timestamp_exact = 0;
      size_t gs_pose_samples_matched_timestamp_interpolated = 0;
      size_t gs_pose_samples_matched_timestamp_nearest = 0;
      size_t gs_pose_samples_missing_pose = 0;
      size_t gs_pose_samples_missing_meta = 0;
      double gs_pose_shift_sum = 0.0;
      double gs_pose_shift_max = 0.0;
      auto rebuildGsCamerasFromFinalPoses = [&]() {
        if (!have_final_poses) {
          return;
        }

        for (auto& gs_pair : gsRuntime_->submapStates) {
          const auto submap_it = seSubmapLookup_.find(gs_pair.first);
          if (submap_it == seSubmapLookup_.end()) {
            continue;
          }
          const Eigen::Isometry3f& T_WK = submap_it->second.T_WK;
          const Eigen::Isometry3f T_KW = T_WK.inverse();

          auto& gs_state = gs_pair.second;
          const size_t n_train =
              std::min(gs_state.gsCamListCpu.size(), gs_state.gtImgListCpu.size());
          if (n_train == 0) {
            continue;
          }

          gs_pose_samples_total += n_train;
          const size_t n_meta =
              std::min(gs_state.poseMetadataCpu.size(), gs_state.gsCamListCpu.size());
          if (n_meta < n_train) {
            gs_pose_samples_missing_meta += (n_train - n_meta);
          }

          for (size_t i = 0; i < n_train; ++i) {
            if (i >= n_meta) {
              continue;
            }
            const auto& pose_meta = gs_state.poseMetadataCpu[i];
            Eigen::Isometry3f T_WS = Eigen::Isometry3f::Identity();
            GsMappingPoseMatch pose_match = GsMappingPoseMatch::None;
            if (!findMappingPoseByTimestampOrState(final_poses_by_state_id,
                                                   final_poses_by_timestamp,
                                                   pose_meta.state_id,
                                                   pose_meta.timestamp_ns,
                                                   T_WS,
                                                   pose_match)) {
              gs_pose_samples_missing_pose++;
              continue;
            }

            gs_pose_samples_matched++;
            switch (pose_match) {
              case GsMappingPoseMatch::ExactTimestamp:
                gs_pose_samples_matched_timestamp_exact++;
                break;
              case GsMappingPoseMatch::InterpolatedTimestamp:
                gs_pose_samples_matched_timestamp_interpolated++;
                break;
              case GsMappingPoseMatch::NearestEndpoint:
                gs_pose_samples_matched_timestamp_nearest++;
                break;
              case GsMappingPoseMatch::StateIdFallback:
                gs_pose_samples_matched_state++;
                break;
              case GsMappingPoseMatch::None:
                break;
            }

            auto& cam_cpu = gs_state.gsCamListCpu[i];
            auto K2C_old = cam_cpu.T_W2C.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto proj_old =
                K2C_old.inverse().mm(cam_cpu.full_proj_matrix.to(torch::kCPU)
                                         .to(torch::kFloat32)
                                         .contiguous())
                    .contiguous();
            auto old_center = K2C_old.inverse()[3].slice(0, 0, 3).contiguous();
            auto old_center_cpu = old_center.to(torch::kCPU).contiguous();
            const float* old_center_ptr = old_center_cpu.data_ptr<float>();
            const Eigen::Vector3f old_center_vec(
                old_center_ptr[0], old_center_ptr[1], old_center_ptr[2]);

            const Eigen::Isometry3f T_SC(pose_meta.T_SC);
            const Eigen::Isometry3f T_WC = T_WS * T_SC;
            const Eigen::Isometry3f T_KC = T_KW * T_WC;
            const Eigen::Matrix4f T_CK = T_KC.inverse().matrix();
            auto K2C_new = torch::from_blob((void*) T_CK.data(), {4, 4}, torch::kFloat32)
                               .clone()
                               .to(torch::kCPU)
                               .contiguous();
            gs::refreshFixedPoseCameraDerivedTensors(cam_cpu, K2C_new, proj_old);

            const Eigen::Vector3f new_center_vec = T_KC.translation();
            const double shift = static_cast<double>(
                (new_center_vec - old_center_vec).norm());
            gs_pose_shift_sum += shift;
            gs_pose_shift_max = std::max(gs_pose_shift_max, shift);
          }
        }

        const double mean_shift =
            gs_pose_samples_matched > 0
                ? gs_pose_shift_sum / static_cast<double>(gs_pose_samples_matched)
                : 0.0;
        LOG(INFO) << std::fixed << std::setprecision(6)
                  << "GS pose rebuild from final provider: total=" << gs_pose_samples_total
                  << " matched=" << gs_pose_samples_matched
                  << " timestamp_exact="
                  << gs_pose_samples_matched_timestamp_exact
                  << " timestamp_interpolated="
                  << gs_pose_samples_matched_timestamp_interpolated
                  << " timestamp_nearest="
                  << gs_pose_samples_matched_timestamp_nearest
                  << " state_fallback=" << gs_pose_samples_matched_state
                  << " missing_pose=" << gs_pose_samples_missing_pose
                  << " missing_meta=" << gs_pose_samples_missing_meta
                  << " mean_shift_m=" << mean_shift
                  << " max_shift_m=" << gs_pose_shift_max;
      };
      rebuildGsCamerasFromFinalPoses();

      for (auto& gs_pair : gsRuntime_->submapStates) {
        auto& gs_state = gs_pair.second;
        if (gs_state.model && gs_state.model->optimizer) {
          gs_state.model->Release_training_state(true);
        }
      }

      LOG(INFO) << "GS final global opt: synchronizing online CUDA work before merge";
      trimGsCudaCache();
      LOG(INFO) << "GS final global opt: reusing any still-resident GPU GT caches "
                   "while keeping cameras on the rebuilt final-pose path.";

      auto getGsSubmapCpuTensors = [&](const uint64_t submap_id,
                                       gs::PlyTensorsCPU* out) -> bool {
        if (!out) {
          return false;
        }
        const auto state_it = gsRuntime_->submapStates.find(submap_id);
        if (state_it == gsRuntime_->submapStates.end()) {
          return false;
        }
        const auto& gs_state = state_it->second;
        if (gs_state.model) {
          *out = gs_state.model->Export_ply_tensors_cpu();
        } else if (gs_state.hasCpuModelSnapshot) {
          *out = gs_state.cpuModelSnapshot;
        } else {
          return false;
        }
        return out->xyz.defined() && out->xyz.numel() > 0;
      };

      auto gsSubmapOutputDirectory = [&](const uint64_t submap_id) {
        const auto state_it = gsRuntime_->submapStates.find(submap_id);
        if (state_it == gsRuntime_->submapStates.end()) {
          return gsRuntime_->outputDirectory / ("submap_" + std::to_string(submap_id));
        }
        const auto& gs_state = state_it->second;
        return gs_state.outputDirectory.empty()
                   ? (gsRuntime_->outputDirectory / ("submap_" + std::to_string(submap_id)))
                   : gs_state.outputDirectory;
      };

      auto saveGsSubmapPly = [&](const uint64_t submap_id,
                                 const gs::PlyTensorsCPU& tensors) {
        if (!tensors.xyz.defined() || tensors.xyz.numel() == 0) {
          return;
        }
        gs::GaussianModel::Save_ply_tensors_cpu(
            tensors, gsSubmapOutputDirectory(submap_id), finalIteration);
      };

      for (auto& gs_pair : gsRuntime_->submapStates) {
        const uint64_t kf_id = gs_pair.first;

        auto submap_it = seSubmapLookup_.find(kf_id);
        if (submap_it == seSubmapLookup_.end()) {
          continue;
        }

        const std::filesystem::path submap_dir =
            gsRuntime_->outputDirectory / ("submap_" + std::to_string(kf_id));
        writeSubmapPose(submap_dir, submap_it->second.T_WK);
      }

      std::vector<torch::Tensor> xyz_list;
      std::vector<torch::Tensor> fdc_list;
      std::vector<torch::Tensor> frest_list;
      std::vector<torch::Tensor> opa_list;
      std::vector<torch::Tensor> scale_list;
      std::vector<torch::Tensor> rot_list;
      std::vector<torch::Tensor> source_submap_id_list;
      std::vector<torch::Tensor> source_row_id_list;

      std::unordered_map<uint64_t, gs::PlyTensorsCPU> final_gs_cpu_snapshots;
      std::unordered_map<uint64_t, GsFisherSnapshotCPU> final_fisher_snapshots;
      std::vector<gs::Camera> merged_cam_cuda;
      std::vector<torch::Tensor> merged_gt_train;
      std::vector<torch::Tensor> merged_depth_target_cpu;
      std::vector<gs::GeometryTarget> merged_resident_geometry_targets;
      std::vector<GsRuntime::GsSamplePoseMetadata,
                  Eigen::aligned_allocator<GsRuntime::GsSamplePoseMetadata>>
          merged_pose_meta_cpu;
      size_t merged_gpu_gt_samples = 0;
      size_t merged_cpu_gt_samples = 0;

      for (auto& gs_pair : gsRuntime_->submapStates) {
        const uint64_t kf_id = gs_pair.first;
        auto& gs_state = gs_pair.second;

        auto submap_it = seSubmapLookup_.find(kf_id);
        if (submap_it == seSubmapLookup_.end()) {
          continue;
        }
        const Eigen::Isometry3f& T_WK = submap_it->second.T_WK;

        gs::PlyTensorsCPU t;
        if (getGsSubmapCpuTensors(kf_id, &t)) {
          final_gs_cpu_snapshots[kf_id] = t;
          if (need_final_fisher_snapshot) {
            GsFisherSnapshotCPU fisher_snapshot;
            fisher_snapshot.submapId = kf_id;
            fisher_snapshot.gaussianCount = t.xyz.size(0);
            fisher_snapshot.sourceCacheVersion = gs_state.fisherCacheVersion;
            fisher_snapshot.historyViewCount = gs_state.fisherHistViewCount;
            fisher_snapshot.estimateUpdateCount = gs_state.fisherEstimateUpdateCount;
            fisher_snapshot.historyEmaAlpha = kGsFisherHistoryEmaAlpha;
            try {
              const auto& fisher_xyz = gs_state.fisherHistDiag.xyz;
              const auto& fisher_opacity = gs_state.fisherHistDiag.opacity;
              if (!fisher_xyz.defined() || fisher_xyz.numel() == 0
                  || !fisher_opacity.defined() || fisher_opacity.numel() == 0) {
                fisher_snapshot.unavailableReason = "fisher_cache_missing";
              } else if (fisher_xyz.dim() != 2 || fisher_xyz.size(0) != t.xyz.size(0)
                         || fisher_xyz.size(1) != 3
                         || fisher_opacity.dim() < 1
                         || fisher_opacity.numel() != t.xyz.size(0)
                         || fisher_opacity.size(0) != t.xyz.size(0)) {
                fisher_snapshot.unavailableReason = "fisher_gaussian_count_mismatch";
              } else {
                fisher_snapshot.xyzDiag =
                    fisher_xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                fisher_snapshot.opacityDiag = fisher_opacity.detach()
                                                    .reshape({t.xyz.size(0), 1})
                                                    .to(torch::kCPU)
                                                    .to(torch::kFloat32)
                                                    .contiguous();
              }
            } catch (const std::exception& e) {
              fisher_snapshot.xyzDiag = torch::Tensor();
              fisher_snapshot.opacityDiag = torch::Tensor();
              fisher_snapshot.unavailableReason =
                  std::string("fisher_cpu_snapshot_failed: ") + e.what();
            }
            final_fisher_snapshots[kf_id] = std::move(fisher_snapshot);
            source_row_id_list.push_back(
                torch::arange(t.xyz.size(0),
                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)));
          }
          xyz_list.push_back(transformPointsToWorld(t.xyz, T_WK).contiguous());

          fdc_list.push_back(t.f_dc.to(torch::kCPU).to(torch::kFloat32).contiguous());
          frest_list.push_back(t.f_rest.to(torch::kCPU).to(torch::kFloat32).contiguous());
          opa_list.push_back(ensureOpacityShape(
              t.opacity.to(torch::kCPU).to(torch::kFloat32)).contiguous());
          scale_list.push_back(t.scale.to(torch::kCPU).to(torch::kFloat32).contiguous());
          rot_list.push_back(transformRotWXYZToWorld(t.rotation, T_WK).contiguous());
          source_submap_id_list.push_back(
              torch::full({t.xyz.size(0)},
                          static_cast<int64_t>(kf_id),
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)));
        }

        const size_t n_train = std::min(gs_state.gsCamListCpu.size(), gs_state.gtImgListCpu.size());
        if (n_train == 0) {
          continue;
        }
        const bool geometry_supervision =
            gsRuntime_->optimParams.geometrySupervisionActive();
        if (geometry_supervision && gs_state.depthTargetListCpu.size() < n_train) {
          throw std::runtime_error(
              "GS global geometry target list is incomplete for submap "
              + std::to_string(kf_id));
        }
        const size_t n_meta =
            std::min(gs_state.poseMetadataCpu.size(), gs_state.gsCamListCpu.size());
        const Eigen::Matrix4f T_KW_e = T_WK.inverse().matrix();
        auto T_KW_cuda = torch::from_blob((void*) T_KW_e.data(), {4, 4}, torch::kFloat32)
                             .clone()
                             .to(torch::kCUDA)
                             .contiguous();

        for (size_t i = 0; i < n_train; ++i) {
          const auto& cam_k = gs_state.gsCamListCpu[i];
          const bool use_gpu_cached_gt = i < gs_state.gtImgList.size();

          gs::Camera cam_w = cam_k;
          torch::Tensor W2C_k = cam_k.T_W2C.to(torch::kCUDA, true).contiguous();
          torch::Tensor proj_k = cam_k.proj_matrix_raw.defined()
                                     ? cam_k.proj_matrix_raw.to(torch::kCUDA, true).contiguous()
                                     : W2C_k.inverse().mm(
                                           cam_k.full_proj_matrix.to(torch::kCUDA, true).contiguous())
                                           .contiguous();
          if (use_gpu_cached_gt) {
            merged_gt_train.push_back(gs_state.gtImgList[i]);
            merged_gpu_gt_samples++;
          } else {
            merged_gt_train.push_back(gs_state.gtImgListCpu[i]);
            merged_cpu_gt_samples++;
          }

          auto T_W2C_world = T_KW_cuda.mm(W2C_k).contiguous();
          gs::refreshFixedPoseCameraDerivedTensors(cam_w, T_W2C_world, proj_k);
          merged_cam_cuda.push_back(std::move(cam_w));
          if (geometry_supervision) {
            merged_depth_target_cpu.push_back(gs_state.depthTargetListCpu[i]);
            if (i < gs_state.geometryTargetList.size()
                && gs_state.geometryTargetList[i].defined()) {
              merged_resident_geometry_targets.push_back(gs_state.geometryTargetList[i]);
            } else {
              merged_resident_geometry_targets.emplace_back();
            }
          }
          if (i < n_meta) {
            merged_pose_meta_cpu.push_back(gs_state.poseMetadataCpu[i]);
          } else {
            merged_pose_meta_cpu.emplace_back();
          }
        }
      }

      if (!xyz_list.empty()) {
        gs::PlyTensorsCPU merged_cpu;
        merged_cpu.xyz = torch::cat(xyz_list, 0).contiguous();
        merged_cpu.f_dc = torch::cat(fdc_list, 0).contiguous();
        merged_cpu.f_rest = torch::cat(frest_list, 0).contiguous();
        merged_cpu.opacity = torch::cat(opa_list, 0).contiguous();
        merged_cpu.scale = torch::cat(scale_list, 0).contiguous();
        merged_cpu.rotation = torch::cat(rot_list, 0).contiguous();
        auto merged_source_submap_ids = torch::cat(source_submap_id_list, 0).contiguous();

        GsFisherDiag merged_preview_fisher;

        torch::Tensor merged_source_row_ids;
        if (need_final_fisher_snapshot) {
          if (source_row_id_list.size() != source_submap_id_list.size()) {
            throw std::runtime_error("GS Fisher row identity list is incomplete before merge");
          }
          merged_source_row_ids = torch::cat(source_row_id_list, 0).contiguous();
        }

        const std::filesystem::path merged_output = gsRuntime_->outputDirectory / "merged";
        gsRuntime_->mergedModel = std::make_unique<gs::GaussianModel>(
            gsRuntime_->optimParams, merged_output.string());
        gsRuntime_->mergedModel->densify_params =
            toGsErrorDensifyParams(gsDensifyConfig_);
        gsRuntime_->mergedModel->Import_ply_tensors_cpu(merged_cpu);
        gs::PlyTensorsCPU final_merged_cpu_snapshot = merged_cpu;

        bool merged_has_gaussians = false;
        {
          auto merged_tensors = gsRuntime_->mergedModel->Export_ply_tensors_cpu();
          merged_has_gaussians = merged_tensors.xyz.defined() && merged_tensors.xyz.numel() > 0;
        }

        PoseLookup gs_optimized_poses_by_timestamp = final_poses_by_timestamp;
        PoseLookup gs_optimized_poses_by_state_id = final_poses_by_state_id;
        bool ran_global_optimization = false;
        bool ran_joint_pose_optimization = false;
        double global_optimization_sec = 0.0;
        global_opt_kf_count = merged_gt_train.size();

        if (merged_has_gaussians
            && submapConfig_.gsRunGlobalOptimization
            && !merged_cam_cuda.empty()
            && !merged_gt_train.empty()) {
          const okvis::Time global_opt_start_time = okvis::Time::now();
          std::vector<torch::Tensor> merged_proj_raw_cuda;
          merged_proj_raw_cuda.reserve(merged_cam_cuda.size());
          for (const auto& cam : merged_cam_cuda) {
            merged_proj_raw_cuda.push_back(cam.proj_matrix_raw);
          }
          LOG(INFO) << "GS final global opt training cache reuse: gpu_gt_samples="
                    << merged_gpu_gt_samples
                    << " cpu_gt_fallback_samples=" << merged_cpu_gt_samples;

          auto& merged_model = *gsRuntime_->mergedModel;
          const bool geometry_supervision =
              merged_model.optimParams.geometrySupervisionActive();
          if (geometry_supervision
              && (merged_depth_target_cpu.size() != merged_cam_cuda.size()
                  || merged_resident_geometry_targets.size() != merged_cam_cuda.size())) {
            throw std::runtime_error(
                "GS global geometry targets are not aligned with the RGB training views");
          }
          const float lambda = merged_model.optimParams.lambda_dssim;
          const int gaussian_only_iters =
              std::max(0, merged_model.optimParams.global_gaussian_only_iters);
          const int joint_pose_iters =
              std::max(0, merged_model.optimParams.global_joint_pose_iters);
          const int total_global_iters = gaussian_only_iters + joint_pose_iters;
          std::unique_ptr<GsGeometryTargetCache> cpu_geometry_cache;
          if (geometry_supervision && total_global_iters > 0) {
            cpu_geometry_cache = std::make_unique<GsGeometryTargetCache>(merged_cam_cuda.size());
          }


          std::vector<GsPoseOptimState> pose_opt_states;
          if (joint_pose_iters > 0) {
            const double kPoseRotLr =
                std::max(0.0, static_cast<double>(merged_model.optimParams.global_pose_rot_lr));
            const double kPoseTransLr = std::max(
                0.0, static_cast<double>(merged_model.optimParams.global_pose_trans_lr));
            LOG(INFO) << "GS joint pose optimization LR: rot=" << kPoseRotLr
                      << " trans=" << kPoseTransLr;
            pose_opt_states.resize(merged_cam_cuda.size());
            for (auto& pose_state : pose_opt_states) {
              pose_state.theta = torch::zeros({3}, torch::TensorOptions()
                                                       .dtype(torch::kFloat32)
                                                       .device(torch::kCUDA))
                                     .set_requires_grad(true);
              pose_state.rho = torch::zeros({3}, torch::TensorOptions()
                                                     .dtype(torch::kFloat32)
                                                     .device(torch::kCUDA))
                                   .set_requires_grad(true);

              std::vector<torch::optim::OptimizerParamGroup> pose_params;
              pose_params.reserve(2);
              pose_params.push_back(torch::optim::OptimizerParamGroup(
                  {pose_state.theta},
                  std::make_unique<torch::optim::AdamOptions>(kPoseRotLr)));
              pose_params.push_back(torch::optim::OptimizerParamGroup(
                  {pose_state.rho},
                  std::make_unique<torch::optim::AdamOptions>(kPoseTransLr)));
              pose_state.optimizer = std::make_unique<torch::optim::Adam>(
                  pose_params, torch::optim::AdamOptions(0.0).eps(1e-15));
            }
          }

          const bool can_export_joint_poses =
              !merged_pose_meta_cpu.empty()
              && merged_pose_meta_cpu.size() == merged_cam_cuda.size();
          auto exportOptimizedPoses = [&](size_t* overwritten_timestamp_count,
                                          size_t* overwritten_state_count) -> size_t {
            if (!can_export_joint_poses) {
              return 0;
            }

            size_t updated_pose_count = 0;
            size_t overwritten_timestamp_local = 0;
            size_t overwritten_state_local = 0;
            for (size_t i = 0; i < merged_cam_cuda.size(); ++i) {
              const auto& pose_meta = merged_pose_meta_cpu[i];
              if (pose_meta.timestamp_ns == 0) {
                continue;
              }

              auto C2W = merged_cam_cuda[i].T_W2C.inverse()
                             .to(torch::kCPU)
                             .to(torch::kFloat32)
                             .contiguous();
              const float* ptr = C2W.data_ptr<float>();

              Eigen::Map<const Eigen::Matrix4f> T_WC_col(ptr);
              Eigen::Isometry3f T_WC = Eigen::Isometry3f::Identity();
              T_WC.matrix() = T_WC_col;

              const Eigen::Isometry3f T_SC(pose_meta.T_SC);
              const Eigen::Isometry3f T_WS = T_WC * T_SC.inverse();
              auto inserted_ts =
                  gs_optimized_poses_by_timestamp.insert_or_assign(pose_meta.timestamp_ns, T_WS);
              if (!inserted_ts.second) {
                overwritten_timestamp_local++;
              }
              if (pose_meta.state_id != 0) {
                auto inserted_state =
                    gs_optimized_poses_by_state_id.insert_or_assign(pose_meta.state_id, T_WS);
                if (!inserted_state.second) {
                  overwritten_state_local++;
                }
              }
              updated_pose_count++;
            }

            if (overwritten_timestamp_count) {
              *overwritten_timestamp_count = overwritten_timestamp_local;
            }
            if (overwritten_state_count) {
              *overwritten_state_count = overwritten_state_local;
            }
            return updated_pose_count;
          };

          auto runGlobalOptIteration = [&](const int iteration_idx,
                                           const bool optimize_pose) {
            const std::vector<int> indices =
                gs::get_random_indices(static_cast<int>(merged_gt_train.size()));
            const int iteration_number = iteration_idx + 1;
            const char* mode = optimize_pose ? "joint-pose" : "gaussian-only";
            LOG(INFO) << "GS global iter " << iteration_number
                      << " render/backward begin: mode=" << mode
                      << " views=" << indices.size();
            int processed_views = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
              const bool first_view = i == 0;
              const bool last_view = i + 1 == indices.size();
              const int idx = indices[i];

              gs::GeometryTarget cpu_geometry_target;
              auto gt = merged_gt_train[idx];
              if (!gt.is_cuda()) {
                gt = gt.to(torch::kCUDA, true);
              }
              if (gt.scalar_type() != torch::kFloat32) {
                gt = gt.to(torch::kFloat32);
              } else if (!gt.is_contiguous()) {
                gt = gt.contiguous();
              }
              auto& cam = merged_cam_cuda[idx];

              torch::Tensor image;
              gs::RenderResult rendered_geometry;
              if (optimize_pose) {
                auto& pose_state = pose_opt_states[idx];
                pose_state.optimizer->zero_grad(true);
                if (geometry_supervision) {
                  rendered_geometry = gs::render_mono_geometry(
                      cam, merged_model, pose_state.theta, pose_state.rho);
                  image = rendered_geometry.rgb;
                } else {
                  std::tie(image, std::ignore, std::ignore, std::ignore) =
                      gs::render_mono_with_pose(cam,
                                                merged_model,
                                                pose_state.theta,
                                                pose_state.rho);
                }
              } else if (geometry_supervision) {
                rendered_geometry = gs::render_faster_geometry(cam, merged_model);
                image = rendered_geometry.rgb;
              } else {
                std::tie(image, std::ignore, std::ignore, std::ignore) =
                    gs::render(cam, merged_model);
              }

              auto l1 = gs::l1_loss(image, gt);
              auto ssim = gs::ssim(image, gt, gs::conv_window, gs::window_size, gs::channel);
              ++processed_views;
              auto loss = (1.0f - lambda) * l1 + lambda * (1.0f - ssim);
              if (geometry_supervision) {
                const auto& resident_target = merged_resident_geometry_targets[idx];
                if (!resident_target.defined()) {
                  cpu_geometry_target = cpu_geometry_cache->get(
                      idx, merged_depth_target_cpu[idx], cam);
                }
                const auto target_cuda = gs::geometryTargetToDevice(
                    resident_target.defined() ? resident_target : cpu_geometry_target,
                    torch::Device(torch::kCUDA));
                loss = loss + gs::geometrySupervisionLoss(
                                  rendered_geometry,
                                  target_cuda,
                                  merged_model.optimParams.lambda_depth,
                                  merged_model.optimParams.lambda_normal)
                                  .total;
              }
              loss.backward();
              if (last_view) {
                LOG(INFO) << "GS global iter " << iteration_number
                          << " render/backward end: mode=" << mode
                          << " processed_views=" << processed_views;
              }

              if (first_view) {
                LOG(INFO) << "GS global iter " << iteration_number
                          << " gaussian-adam begin: mode=" << mode;
              }
              gs::fused_adam_step(merged_model.optimizer.get());
              merged_model.optimizer->zero_grad(true);
              if (last_view) {
                LOG(INFO) << "GS global iter " << iteration_number
                          << " gaussian-adam end: mode=" << mode;
              }

              if (optimize_pose) {
                auto& pose_state = pose_opt_states[idx];
                if (first_view) {
                  LOG(INFO) << "GS global iter " << iteration_number
                            << " pose-adam begin";
                }
                pose_state.optimizer->step();
                applyPoseDeltaToCamera(
                    cam, pose_state.rho, pose_state.theta, merged_proj_raw_cuda[idx]);
                pose_state.theta.data().zero_();
                pose_state.rho.data().zero_();
                pose_state.optimizer->zero_grad(true);
                if (last_view) {
                  LOG(INFO) << "GS global iter " << iteration_number
                            << " pose-adam end";
                }
              }

              if (submapConfig_.gsShowPreview && i + 1 == indices.size()) {
                {
                  gs::DataPacket packet;
                  packet.ID = (integration_counter_ > 0)
                                  ? static_cast<int>(integration_counter_ - 1)
                                  : -1;
                  packet.global_iter = iteration_number;
                  packet.num_splats = merged_model.Get_size();
                  packet.num_kf = static_cast<int>(merged_gt_train.size());
                  packet.fps = 0.0f;
                  try {
                    auto rendered_rgb_hwc =
                        image.detach().permute({1, 2, 0}).contiguous();
                    packet.rendered_rgb_staging =
                        gs::stagePreviewRgbHwcFloatToCpu(rendered_rgb_hwc);
                  } catch (const std::exception& error) {
                    LOG(WARNING)
                        << "GS global rendered preview staging dropped iteration "
                        << iteration_number << ": " << error.what();
                  }
                  if (packet.rendered_rgb_staging.valid()) {
                    gsRuntime_->dataQueue.push(std::move(packet));
                  }
                }
              }
            }

          };

          size_t last_updated_pose_count = 0;
          size_t last_overwritten_timestamp_count = 0;
          size_t last_overwritten_state_count = 0;
          int iter_counter = 0;
          for (int it = 0; it < gaussian_only_iters; ++it, ++iter_counter) {
            runGlobalOptIteration(iter_counter, false);

            std::cout << "\rGS global optimization [gaussian-only]: "
                      << (iter_counter + 1) << "/" << total_global_iters
                      << "  splats=" << merged_model.Get_size()
                      << "  kf=" << merged_gt_train.size()
                      << std::flush;
          }
          for (int it = 0; it < joint_pose_iters; ++it, ++iter_counter) {
            runGlobalOptIteration(iter_counter, true);
            last_updated_pose_count = exportOptimizedPoses(
                &last_overwritten_timestamp_count, &last_overwritten_state_count);
            if (last_updated_pose_count > 0) {
              ran_joint_pose_optimization = true;
              publishGsOptimizedTrajectory(gs_optimized_poses_by_state_id);
            }

            std::cout << "\rGS global optimization [joint-pose]: "
                      << (iter_counter + 1) << "/" << total_global_iters
                      << "  splats=" << merged_model.Get_size()
                      << "  kf=" << merged_gt_train.size()
                      << std::flush;
          }
          if (total_global_iters > 0) {
            std::cout << std::endl;
          }

          if (joint_pose_iters > 0) {
            if (can_export_joint_poses) {
              LOG(INFO) << "GS joint pose optimization updated poses: updated="
                        << last_updated_pose_count
                        << " overwritten_timestamp=" << last_overwritten_timestamp_count
                        << " overwritten_state=" << last_overwritten_state_count;
            } else {
              LOG(WARNING) << "GS joint pose optimization could not export optimized poses "
                           << "(meta_size=" << merged_pose_meta_cpu.size()
                           << ", cams=" << merged_cam_cuda.size() << ")";
            }
          }

          {
            const int device_index = merged_model.Get_xyz().get_device();
            c10::cuda::CUDAGuard device_guard(device_index);
            c10::cuda::getCurrentCUDAStream(device_index).synchronize();
          }
          if (cpu_geometry_cache) {
            LOG(INFO) << "GS global geometry CPU cache: views="
                      << cpu_geometry_cache->cachedViews()
                      << " resident_targets="
                      << std::count_if(merged_resident_geometry_targets.begin(),
                                       merged_resident_geometry_targets.end(),
                                       [](const gs::GeometryTarget& target) { return target.defined(); })
                      << " payload_mib="
                      << cpu_geometry_cache->payloadBytes() / (1024.0 * 1024.0);

            cpu_geometry_cache.reset();
          }

          merged_model.Release_training_state(false);

          global_optimization_sec = (okvis::Time::now() - global_opt_start_time).toSec();
          ran_global_optimization = total_global_iters > 0;

          if (ran_global_optimization) {
            const auto merged_tensors_cpu = gsRuntime_->mergedModel->Export_ply_tensors_cpu();
            final_merged_cpu_snapshot = merged_tensors_cpu;
            const auto source_ids_cpu = merged_source_submap_ids.to(torch::kCPU).contiguous();
            const auto source_ids_flat = source_ids_cpu.view({-1});
            const auto source_row_ids_flat =
                merged_source_row_ids.defined()
                    ? merged_source_row_ids.to(torch::kCPU).contiguous().view({-1})
                    : torch::Tensor();
            const bool fisher_row_identity_valid =
                !need_final_fisher_snapshot
                || (source_row_ids_flat.defined()
                    && source_row_ids_flat.numel() == source_ids_flat.numel()
                    && source_row_ids_flat.numel() == merged_tensors_cpu.xyz.size(0));
            if (need_final_fisher_snapshot && !fisher_row_identity_valid) {
              LOG(ERROR) << "GS Fisher row identity is unavailable after global optimization; "
                            "final Fisher will be marked unavailable.";
            }
            const auto cpu_float_opts =
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            size_t updated_submaps = 0;

            for (auto& gs_pair : gsRuntime_->submapStates) {
              const uint64_t kf_id = gs_pair.first;
              auto& gs_state = gs_pair.second;
              const auto submap_it = seSubmapLookup_.find(kf_id);
              if (submap_it == seSubmapLookup_.end()) {
                continue;
              }

              gs::PlyTensorsCPU current_tensors;
              auto current_snapshot_it = final_gs_cpu_snapshots.find(kf_id);
              bool had_gaussians = current_snapshot_it != final_gs_cpu_snapshots.end();
              if (had_gaussians) {
                current_tensors = current_snapshot_it->second;
              } else {
                had_gaussians = getGsSubmapCpuTensors(kf_id, &current_tensors);
              }
              auto indices =
                  torch::nonzero(source_ids_flat == static_cast<int64_t>(kf_id)).view({-1});
              if (!had_gaussians && indices.numel() == 0) {
                continue;
              }

              gs::PlyTensorsCPU local_tensors;
              if (indices.numel() > 0) {
                local_tensors.xyz = transformPointsToSubmap(
                                        merged_tensors_cpu.xyz.index_select(0, indices),
                                        submap_it->second.T_WK)
                                        .contiguous();
                local_tensors.f_dc =
                    merged_tensors_cpu.f_dc.index_select(0, indices).contiguous();
                local_tensors.f_rest =
                    merged_tensors_cpu.f_rest.index_select(0, indices).contiguous();
                local_tensors.opacity =
                    ensureOpacityShape(
                        merged_tensors_cpu.opacity.index_select(0, indices).contiguous())
                        .contiguous();
                local_tensors.scale =
                    merged_tensors_cpu.scale.index_select(0, indices).contiguous();
                local_tensors.rotation = transformRotWXYZToSubmap(
                                             merged_tensors_cpu.rotation.index_select(0, indices),
                                             submap_it->second.T_WK)
                                             .contiguous();
              } else {
                const int64_t f_rest_dim =
                    current_tensors.f_rest.defined() && current_tensors.f_rest.dim() == 2
                        ? current_tensors.f_rest.size(1)
                        : 0;
                local_tensors.xyz = torch::empty({0, 3}, cpu_float_opts);
                local_tensors.f_dc = torch::empty({0, 3}, cpu_float_opts);
                local_tensors.f_rest = torch::empty({0, f_rest_dim}, cpu_float_opts);
                local_tensors.opacity = torch::empty({0, 1}, cpu_float_opts);
                local_tensors.scale = torch::empty({0, 3}, cpu_float_opts);
                local_tensors.rotation = torch::empty({0, 4}, cpu_float_opts);
              }

              if (need_final_fisher_snapshot) {
                auto fisher_it = final_fisher_snapshots.find(kf_id);
                if (fisher_it == final_fisher_snapshots.end()) {
                  GsFisherSnapshotCPU missing_snapshot;
                  missing_snapshot.submapId = kf_id;
                  missing_snapshot.unavailableReason = "fisher_snapshot_missing_before_global_opt";
                  fisher_it = final_fisher_snapshots.emplace(kf_id, std::move(missing_snapshot)).first;
                }
                auto& fisher_snapshot = fisher_it->second;
                fisher_snapshot.contentRevision = 2;
                fisher_snapshot.gaussianCount = local_tensors.xyz.size(0);
                fisher_snapshot.evaluatedStage = "pre_global_optimization";
                fisher_snapshot.alignedStage = "post_global_optimization";
                fisher_snapshot.recomputedAfterGlobalOptimization = false;

                if (fisher_snapshot.available()) {
                  if (!fisher_row_identity_valid) {
                    fisher_snapshot.xyzDiag = torch::Tensor();
                    fisher_snapshot.opacityDiag = torch::Tensor();
                    fisher_snapshot.unavailableReason = "fisher_row_identity_unavailable";
                  } else {
                    auto original_rows = source_row_ids_flat.index_select(0, indices)
                                             .to(torch::kLong)
                                             .contiguous();
                    const bool row_count_matches =
                        original_rows.numel() == local_tensors.xyz.size(0);
                    const bool rows_in_bounds =
                        original_rows.numel() == 0
                        || (original_rows.min().item<int64_t>() >= 0
                            && original_rows.max().item<int64_t>()
                                   < fisher_snapshot.xyzDiag.size(0));
                    if (!row_count_matches || !rows_in_bounds) {
                      fisher_snapshot.xyzDiag = torch::Tensor();
                      fisher_snapshot.opacityDiag = torch::Tensor();
                      fisher_snapshot.unavailableReason = "fisher_row_identity_out_of_bounds";
                    } else {
                      fisher_snapshot.xyzDiag =
                          fisher_snapshot.xyzDiag.index_select(0, original_rows).contiguous();
                      fisher_snapshot.opacityDiag =
                          fisher_snapshot.opacityDiag.index_select(0, original_rows).contiguous();
                    }
                  }
                }
              }

              if (gs_state.model) {
                gs_state.model->Import_ply_tensors_cpu(local_tensors);
                gs_state.model->Release_training_state(true);
              } else {
                gs_state.cpuModelSnapshot = local_tensors;
                gs_state.hasCpuModelSnapshot = true;
              }
              final_gs_cpu_snapshots[kf_id] = local_tensors;
              invalidateGsSubmapFisherCache(kf_id);
              const auto fisher_it = final_fisher_snapshots.find(kf_id);
              if (fisher_it != final_fisher_snapshots.end() && fisher_it->second.available()) {
                const auto& fisher_snapshot = fisher_it->second;
                gs_state.fisherHistDiag.xyz = fisher_snapshot.xyzDiag;
                gs_state.fisherHistDiag.opacity = fisher_snapshot.opacityDiag;
                gs_state.fisherHistViewCount = fisher_snapshot.historyViewCount;
                gs_state.fisherEstimateUpdateCount = fisher_snapshot.estimateUpdateCount;
              }
              updated_submaps++;
            }

            updateGsSubmapResidencyPolicy(prevKeyframeId_);
            waitForGsModelOffloads();
            LOG(INFO) << "GS global opt write-back updated " << updated_submaps
                      << " source submaps.";
          }
        }

        if (ran_joint_pose_optimization) {
          final_poses_by_timestamp = gs_optimized_poses_by_timestamp;
          final_poses_by_state_id = gs_optimized_poses_by_state_id;
          publishGsOptimizedTrajectory(final_poses_by_state_id);
          joint_pose_applied = true;
        }

        global_optimization_sec_summary = global_optimization_sec;

        if (ran_global_optimization) {
          LOG(INFO) << std::fixed << std::setprecision(3)
                    << "GS global optimization time: " << global_optimization_sec << " s";
        } else {
          LOG(INFO) << "GS global optimization skipped "
                    << "(enabled=" << submapConfig_.gsRunGlobalOptimization
                    << ", has_gaussians=" << merged_has_gaussians
                    << ", train_cams=" << merged_cam_cuda.size()
                    << ", train_rgb=" << merged_gt_train.size() << ")";
        }

        GsFisherExportOptions fisher_export_options;
        fisher_export_options.xyzWeight = gsFisherScoreConfig_.weight_xyz;
        fisher_export_options.opacityWeight = gsFisherScoreConfig_.weight_opacity;
        fisher_export_options.eigRegularizer =
            occupancyViewerConfig_.fisher_eig_regularizer;
        fisher_export_options.displayRegularizer =
            occupancyViewerConfig_.fisher_display_regularizer;

        for (auto& gs_pair : gsRuntime_->submapStates) {
          gs::PlyTensorsCPU tensors;
          auto tensor_it = final_gs_cpu_snapshots.find(gs_pair.first);
          bool have_tensors = tensor_it != final_gs_cpu_snapshots.end();
          if (have_tensors) {
            tensors = tensor_it->second;
          } else {
            have_tensors = getGsSubmapCpuTensors(gs_pair.first, &tensors);
          }
          if (have_tensors) {
            saveGsSubmapPly(gs_pair.first, tensors);
            if (submapConfig_.gsSaveFisher) {
              auto fisher_it = final_fisher_snapshots.find(gs_pair.first);
              if (fisher_it == final_fisher_snapshots.end()) {
                GsFisherSnapshotCPU missing_snapshot;
                missing_snapshot.submapId = gs_pair.first;
                missing_snapshot.gaussianCount = tensors.xyz.size(0);
                missing_snapshot.contentRevision = ran_global_optimization ? 2 : 1;
                missing_snapshot.unavailableReason = "fisher_snapshot_missing";
                fisher_it = final_fisher_snapshots.emplace(
                    gs_pair.first, std::move(missing_snapshot)).first;
              }

              auto& fisher_snapshot = fisher_it->second;
              fisher_snapshot.contentRevision = ran_global_optimization ? 2 : 1;
              fisher_snapshot.gaussianCount = tensors.xyz.size(0);
              if (!ran_global_optimization) {
                fisher_snapshot.evaluatedStage = "final_online_state";
                fisher_snapshot.alignedStage = "same_content";
              } else if (fisher_snapshot.alignedStage
                         != "post_global_optimization") {
                fisher_snapshot.xyzDiag = torch::Tensor();
                fisher_snapshot.opacityDiag = torch::Tensor();
                fisher_snapshot.unavailableReason = "fisher_global_alignment_not_performed";
              }

              if (fisher_snapshot.available()
                  && (fisher_snapshot.xyzDiag.dim() != 2
                      || fisher_snapshot.xyzDiag.size(0) != tensors.xyz.size(0)
                      || fisher_snapshot.xyzDiag.size(1) != 3
                      || fisher_snapshot.opacityDiag.numel() != tensors.xyz.size(0))) {
                fisher_snapshot.xyzDiag = torch::Tensor();
                fisher_snapshot.opacityDiag = torch::Tensor();
                fisher_snapshot.unavailableReason = "fisher_final_gaussian_count_mismatch";
              }

              const std::filesystem::path iteration_directory =
                  gsSubmapOutputDirectory(gs_pair.first)
                  / ("iteration_" + std::to_string(finalIteration));
              const std::filesystem::path gs_ply_path =
                  iteration_directory / "point_cloud.ply";
              std::string fisher_error;
              const bool wrote_fisher = writeGsFisherSnapshot(iteration_directory,
                                                               gs_ply_path,
                                                               tensors,
                                                               fisher_snapshot,
                                                               fisher_export_options,
                                                               &fisher_error);
              if (!wrote_fisher) {
                fisher_export_ok = false;
                fisher_submaps_unavailable++;
                LOG(ERROR) << "Failed to export GS Fisher for submap " << gs_pair.first
                           << ": " << fisher_error;
              } else if (fisher_snapshot.available()) {
                fisher_submaps_available++;
                LOG(INFO) << "Saved aligned GS Fisher for submap " << gs_pair.first
                          << " (revision=" << fisher_snapshot.contentRevision
                          << ", gaussians=" << fisher_snapshot.gaussianCount << ").";
              } else {
                fisher_export_ok = false;
                fisher_submaps_unavailable++;
                LOG(WARNING) << "GS Fisher unavailable for submap " << gs_pair.first
                             << ": " << fisher_snapshot.unavailableReason;
              }
            }
          }

        }
        if (submapConfig_.gsSaveFisher) {
          LOG(INFO) << "GS Fisher export summary: available=" << fisher_submaps_available
                    << " unavailable=" << fisher_submaps_unavailable;
        }

        if (have_final_poses && !finalTrajectoryCsv.empty()) {
          const std::filesystem::path gs_trajectory_csv =
              buildGsOptimizedTrajectoryPath(finalTrajectoryCsv);
          size_t replaced_rows = 0;
          if (writeGsOptimizedTrajectoryCsv(finalTrajectoryCsv,
                                            gs_optimized_poses_by_timestamp,
                                            gs_trajectory_csv,
                                            &replaced_rows)) {
            joint_pose_updated_rows = replaced_rows;
            LOG(INFO) << "GS finalization: wrote optimized trajectory CSV to "
                      << gs_trajectory_csv << " (replaced_rows=" << replaced_rows << ")";
          } else {
            LOG(WARNING) << "GS finalization: failed to write optimized trajectory CSV from "
                         << finalTrajectoryCsv;
          }
        }

        if (merged_has_gaussians) {
          gs::GaussianModel::Save_ply_tensors_cpu(
              final_merged_cpu_snapshot,
              gsRuntime_->mergedModel->output_path,
              finalIteration);
        }
      } else {
        LOG(WARNING) << "GS finalization: no gaussian submap available for merged export.";
      }

      if (submapConfig_.gsSaveRenderedRgb) {
        torch::NoGradGuard final_render_no_grad;
        if (gsRuntime_->mergedModel
            && gsRuntime_->mergedModel->optimizer
            && gsRuntime_->mergedModel->Get_size() > 0) {
          const int device_index =
              gsRuntime_->mergedModel->Get_xyz().get_device();
          c10::cuda::CUDAGuard device_guard(device_index);
          c10::cuda::getCurrentCUDAStream(device_index).synchronize();
          gsRuntime_->mergedModel->Release_training_state(false);
        }
        gs::GaussianModel* render_model = nullptr;
        if (gsRuntime_->mergedModel
            && gsRuntime_->mergedModel->Get_size() > 0) {
          render_model = gsRuntime_->mergedModel.get();
        }
        if (!render_model) {
          for (auto& gs_pair : gsRuntime_->submapStates) {
            auto& gs_state = gs_pair.second;
            if (gs_state.model && gs_state.model->Get_size() > 0) {
              render_model = gs_state.model.get();
              break;
            }
          }
        }

        if (!render_model) {
          LOG(WARNING) << "GS finalization: rendered RGB export skipped (no GS model).";
        } else if (!cameraSensors_) {
          LOG(WARNING) << "GS finalization: rendered RGB export skipped (no camera sensors).";
        } else if (!have_final_poses) {
          LOG(WARNING) << "GS finalization: rendered RGB export skipped (final trajectory missing or empty).";
        } else if (!have_rgb_frames) {
          LOG(WARNING) << "GS finalization: rendered RGB export skipped "
                          "(no provided frame index and "
                       << rgb_stream << "/data.csv missing or empty).";
        } else {
          size_t colour_cam_idx = std::numeric_limits<size_t>::max();
          if (viParameters_.mappingInputPreprocessor
              && viParameters_.mappingInputPreprocessor->enabled()) {
            colour_cam_idx =
                viParameters_.mappingInputPreprocessor->targetCameraIndex();
          } else {
            for (const auto& sensor_it : *cameraSensors_) {
              if (viParameters_.nCameraSystem.cameraType(sensor_it.first).isColour) {
                colour_cam_idx = sensor_it.first;
                break;
              }
            }
          }
          if (colour_cam_idx == std::numeric_limits<size_t>::max()) {
            LOG(WARNING) << "GS finalization: no colour camera for rendered RGB export.";
          } else {
            auto sensor_it = cameraSensors_->find(colour_cam_idx);
            if (sensor_it == cameraSensors_->end()) {
              LOG(WARNING) << "GS finalization: colour camera sensor missing.";
            } else {
              const auto& colour_sensor = sensor_it->second.second;
              kinematics::Transformation T_SC =
                  *viParameters_.mappingT_SC(colour_cam_idx);
              const Eigen::Isometry3f T_SC_f(T_SC.T().cast<float>());
              auto proj_matrix = gs::getProjectionMatrix(
                                     colour_sensor.model.imageWidth(),
                                     colour_sensor.model.imageHeight(),
                                     colour_sensor.model.focalLengthU(),
                                     colour_sensor.model.focalLengthV(),
                                     colour_sensor.model.imageCenterU(),
                                     colour_sensor.model.imageCenterV())
                                     .to(torch::kCUDA, true);

              std::filesystem::create_directories(render_output_directory);

              size_t skipped_empty_render = 0;
              const size_t total = rgb_frames.size();
              for (size_t i = 0; i < rgb_frames.size(); ++i) {
                const auto& rgb_entry = rgb_frames[i];
                Eigen::Isometry3f T_WS = Eigen::Isometry3f::Identity();
                bool have_render_pose = false;
                {
                  have_render_pose = findNearestPose(
                      final_poses_by_timestamp, rgb_entry.first, T_WS);
                }
                if (!have_render_pose) {
                  ++render_pose_miss_count;
                  std::cout << "\rGS rendered RGB: " << (i + 1) << "/" << total
                            << "  saved=" << render_saved_count << " (pose-miss)"
                            << std::flush;
                  continue;
                }

                const Eigen::Isometry3f T_WC = T_WS * T_SC_f;
                const Eigen::Matrix4f T_CW = T_WC.inverse().matrix();
                auto W2C = torch::from_blob((void*) T_CW.data(), {4, 4}, torch::kFloat32)
                               .clone()
                               .to(torch::kCUDA, true);

                gs::Camera cam;
                cam.width = colour_sensor.model.imageWidth();
                cam.height = colour_sensor.model.imageHeight();
                cam.fov_x = colour_sensor.horizontal_fov;
                cam.fov_y = colour_sensor.vertical_fov;
                cam.fx = colour_sensor.model.focalLengthU();
                cam.fy = colour_sensor.model.focalLengthV();
                cam.cx = colour_sensor.model.imageCenterU();
                cam.cy = colour_sensor.model.imageCenterV();
                gs::refreshFixedPoseCameraDerivedTensors(cam, W2C, proj_matrix);

                torch::Tensor image;
                std::tie(image, std::ignore, std::ignore, std::ignore) =
                    gs::render(cam, *render_model);
                const bool valid_render =
                    image.defined()
                    && image.dim() == 3
                    && image.size(0) == 3
                    && image.size(1) > 0
                    && image.size(2) > 0
                    && image.numel() > 0;
                if (!valid_render) {
                  ++skipped_empty_render;
                  std::cout << "\rGS rendered RGB: " << (i + 1) << "/" << total
                            << "  saved=" << render_saved_count
                            << " (empty-render)"
                            << std::flush;
                  continue;
                }
                auto rendered_tensor =
                    image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
                rendered_tensor =
                    rendered_tensor.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat rgb_u8(
                    image.size(1), image.size(2), CV_8UC3, rendered_tensor.data_ptr<uint8_t>());
                if (rgb_u8.empty()) {
                  ++skipped_empty_render;
                  std::cout << "\rGS rendered RGB: " << (i + 1) << "/" << total
                            << "  saved=" << render_saved_count
                            << " (empty-rgb)"
                            << std::flush;
                  continue;
                }
                cv::Mat bgr_u8;
                cv::cvtColor(rgb_u8, bgr_u8, cv::COLOR_RGB2BGR);

                const std::filesystem::path out_path =
                    render_output_directory / rgb_entry.second;
                if (!out_path.parent_path().empty()) {
                  std::filesystem::create_directories(out_path.parent_path());
                }
                if (cv::imwrite(out_path.string(), bgr_u8)) {
                  ++render_saved_count;
                }
                std::cout << "\rGS rendered RGB: " << (i + 1) << "/" << total
                          << "  saved=" << render_saved_count
                          << std::flush;
              }
              std::cout << std::endl;
              LOG(INFO) << "GS finalization: saved rendered RGB frames "
                        << render_saved_count << " to " << render_output_directory
                        << ", pose misses " << render_pose_miss_count
                        << ", skipped empty renders " << skipped_empty_render;
            }
          }
        }
      }

      size_t final_global_gaussian_count = 0;
      if (gsRuntime_->mergedModel) {
        final_global_gaussian_count =
            static_cast<size_t>(gsRuntime_->mergedModel->Get_size());
      }

      double gpu_memory_usage_mb = -1.0;
      size_t free_gpu_bytes = 0;
      size_t total_gpu_bytes = 0;
      const cudaError_t cuda_mem_status = cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes);
      if (cuda_mem_status == cudaSuccess && total_gpu_bytes > 0) {
        const double gpu_used_memory_end_mb =
            static_cast<double>(total_gpu_bytes - free_gpu_bytes) / (1024.0 * 1024.0);
        if (gsGpuUsedMemoryStartMb_ >= 0.0) {
          gpu_memory_usage_mb =
              std::max(0.0, gpu_used_memory_end_mb - gsGpuUsedMemoryStartMb_);
        } else {
          LOG(WARNING) << "GS finalization: GPU start memory unavailable, cannot compute end-start usage.";
        }
      } else {
        LOG(WARNING) << "GS finalization: cudaMemGetInfo failed: "
                     << cudaGetErrorString(cuda_mem_status);
      }

      {
        std::lock_guard<std::mutex> lock(gsFinalizationSummaryMtx_);
        lastGsFinalizationSummary_.valid = true;
        lastGsFinalizationSummary_.pre_global_opt_system_fps = pre_global_opt_system_fps;
        lastGsFinalizationSummary_.global_optimization_time_sec =
            global_optimization_sec_summary;
        lastGsFinalizationSummary_.global_opt_kf_count = global_opt_kf_count;
        lastGsFinalizationSummary_.save_iteration_frames = final_iteration_raw;
        lastGsFinalizationSummary_.final_global_gaussian_count =
            final_global_gaussian_count;
        lastGsFinalizationSummary_.gpu_memory_usage_mb = gpu_memory_usage_mb;
        lastGsFinalizationSummary_.render_requested = render_requested;
        lastGsFinalizationSummary_.render_frame_count = rgb_frames.size();
        lastGsFinalizationSummary_.render_saved_count = render_saved_count;
        lastGsFinalizationSummary_.render_pose_miss_count = render_pose_miss_count;
        lastGsFinalizationSummary_.render_output_directory =
            render_output_directory.string();
        lastGsFinalizationSummary_.joint_pose_applied = joint_pose_applied;
        lastGsFinalizationSummary_.joint_pose_updated_rows =
            joint_pose_updated_rows;
      }

      LOG(INFO) << "GS finalization completed.";

      return !submapConfig_.gsSaveFisher || fisher_export_ok;
    } catch (const std::exception& e) {

      LOG(ERROR) << "GS finalization failed: " << e.what();
      return false;
    }
  }

  float calculateVolume(const Eigen::Vector3f& mins, const Eigen::Vector3f& maxs) {
    return std::max(0.0f, maxs.x() - mins.x()) *
           std::max(0.0f, maxs.y() - mins.y()) *
           std::max(0.0f, maxs.z() - mins.z());
  }

  float calculateIntersectionVolume(const Eigen::Vector3f& mins1, const Eigen::Vector3f& maxs1,
                                    const Eigen::Vector3f& mins2, const Eigen::Vector3f& maxs2) {
    float x_overlap = std::max(0.0f, std::min(maxs1.x(), maxs2.x()) - std::max(mins1.x(), mins2.x()));
    float y_overlap = std::max(0.0f, std::min(maxs1.y(), maxs2.y()) - std::max(mins1.y(), mins2.y()));
    float z_overlap = std::max(0.0f, std::min(maxs1.z(), maxs2.z()) - std::max(mins1.z(), mins2.z()));
    return x_overlap * y_overlap * z_overlap;
  }

  float calculateIoU3D(const Eigen::Vector3f& mins1, const Eigen::Vector3f& maxs1,
                       const Eigen::Vector3f& mins2, const Eigen::Vector3f& maxs2) {
    float intersection_volume = calculateIntersectionVolume(mins1, maxs1, mins2, maxs2);
    float union_volume = calculateVolume(mins1, maxs1) + calculateVolume(mins2, maxs2) - intersection_volume;

    if (union_volume < std::numeric_limits<float>::epsilon()) {
      return 0.0f;
    }

    return intersection_volume / union_volume;
  }

    bool SubmappingInterface::addDepthMeasurement(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        sensorMeasurementDownsamplingCounter_++;

        if(sensorMeasurementDownsampling_ == 0){
          throw std::runtime_error("The sensorMeasurementDownsampling_ is set to 0. This seems to be an error in your config file");
        }
        if(sensorMeasurementDownsamplingCounter_ % sensorMeasurementDownsampling_ == 0){
          sensorMeasurementDownsamplingCounter_ = 0;

          for (auto& frame : frames) {
            for (auto& depthFrame : frame.second) {
              depthFrame.timeStamp = depthFrame.timeStamp - Duration(viParameters_.camera.image_delay);
            }
          }
          DepthMeasurementPtr frames_ptr =
              std::make_shared<DepthMeasurementMap>(std::move(frames));

          if (depthQueueDropIfFull_) {
            const bool dropped =
                depthMeasurements_.PushNonBlockingDroppingIfFull(frames_ptr, depthQueueSize_);
            if (dropped) {
              recordLatestWinsDrops("mapping-depth queue entries",
                                    realtimeDepthQueueDropCount_, 1);
            }
            return true;
          }

          if(blocking_){
            const bool result = depthMeasurements_.PushBlockingIfFull(frames_ptr, depthQueueSize_);
            return result;
          }else{
            const bool dropped =
                depthMeasurements_.PushNonBlockingDroppingIfFull(frames_ptr, depthQueueSize_);
            if(dropped) {
              recordLatestWinsDrops("mapping-depth queue entries",
                                    realtimeDepthQueueDropCount_, 1);
            }
            return true;
          }
        }

        return true;

    }


    void SubmappingInterface::integrationLoop(){

      LOG(INFO) << "In the integration loop";
      while(!isFinished()){
        checkForAvailableData();
      }
    }

    bool SubmappingInterface::finishedIntegrating(){
      const uint64_t supereightQueueSize = supereightFrames_.Size();
      const uint64_t stateQueueSize = stateUpdates_.Size();
      uint64_t depthQueueSize = depthMeasurements_.Size();
      uint64_t lidarQueueSize = lidarMeasurements_.Size();

      if(supereightQueueSize == 0 && stateQueueSize == 0
         && seSubmapLookup_.size() > 0 && !isProcessingSeFrame_){

        if(frame_){
          const size_t flushQueueSize =
              submapConfig_.realtimePipelineEnabled ? nonBlockingSupereightQueueSize_ : size_t(2);

          const bool dropped = supereightFrames_.PushNonBlockingDroppingIfFull(
              *frame_, flushQueueSize);
          if (dropped) {
            recordLatestWinsDrops("Supereight queue entries",
                                  realtimeSupereightQueueDropCount_, 1);
          }
          frame_.reset();
          return false;
        }

        if(depthQueueSize + lidarQueueSize > 0){
          finishedIntegratingSensorOnlyCycles_++;
          if(finishedIntegratingSensorOnlyCycles_ <= 3){
            return false;
          }
          LOG(WARNING) << "Dropping stale sensor queues at shutdown: depth="
                       << depthQueueSize << " lidar=" << lidarQueueSize;
          while(!depthMeasurements_.Empty()) {
            DepthMeasurementPtr depthMeasurement;
            depthMeasurements_.PopBlocking(&depthMeasurement);
          }
          while(!lidarMeasurements_.Empty()) {
            LidarMeasurement oldestLidarMeasurement;
            lidarMeasurements_.PopBlocking(&oldestLidarMeasurement);
          }
        } else {
          finishedIntegratingSensorOnlyCycles_ = 0;
        }

        if (submapConfig_.useMap2MapFactors && submapCounter_ > 2) {
          if (cameraSensors_) {
            depthMap2MapFactors();
          }
        }

        if(prevKeyframeId_ != UNINITIALIZED_ID){
          enqueueSubmapFinalization(prevKeyframeId_);
        }

        return true;
      }

      finishedIntegratingSensorOnlyCycles_ = 0;
      return false;
    }

    void SubmappingInterface::frameUpdater(okvis::OkvisUpdate& propagatedStateUpdate) {

      if (propagatedStateUpdate.trackingState.isKeyframe || !latestKeyframe.started) { 
        latestKeyframe.started = true;

        if(propagatedStateUpdate.latestState.id.value() != latestKeyframe.Id) {
          latestKeyframe.Id = propagatedStateUpdate.latestState.id.value();
        }
      }
      siFrameDataMap frameDataMap;

      for (const auto& updatedState : *propagatedStateUpdate.updatedStates) {
        frameDataMap[updatedState.first.value()] = SiFrameData(updatedState.first.value(), updatedState.second.T_WS, updatedState.second.covisibleFrameIds);
      }

      bool loop_closure = propagatedStateUpdate.trackingState.recognisedPlace;

      if(!frame_) {
        frame_.reset(new SupereightFrames(latestKeyframe.Id, frameDataMap, loop_closure));
      }
      else {
        frame_->loop_closure = loop_closure;
        updateSEFrameData(frame_->frameData_, frameDataMap);
      }
    }

    bool SubmappingInterface::predict(const okvis::Time &finalTimestamp,
                                      const Transformation &T_SC,
                                      Transformation &T_WC,
                                      StateId* sourceStateId)
    {

      okvis::State measurementState;

      if (gtPoses_) {
        measurementState.T_WS = gtPoses_->at(finalTimestamp.toNSec());
      } else {
        while(trajectoryLocked_);
        trajectoryLocked_ = true;
        if(!propagatedStates.getState(finalTimestamp, measurementState)){
          trajectoryLocked_ = false;
          return false;
        }
        trajectoryLocked_ = false;
      }

      if (sourceStateId) {
        *sourceStateId = measurementState.id;
      }
      T_WC = measurementState.T_WS * T_SC;
      return true;

    }

    void SubmappingInterface::saveSubmap(uint64_t kfId){
      std::string mesh_numbering;
      if (kfId < 10) {
        mesh_numbering = "00000" + std::to_string(kfId);
      }
      else if (kfId < 100) {
        mesh_numbering = "0000" + std::to_string(kfId);
      }
      else if (kfId < 1000) {
        mesh_numbering = "000" + std::to_string(kfId);
      }
      else if (kfId < 10000) {
        mesh_numbering = "00" + std::to_string(kfId);
      }
      else if (kfId < 100000) {
        mesh_numbering = "0" + std::to_string(kfId);
      }
      else if (kfId < 1000000) {
        mesh_numbering = std::to_string(kfId);
      }

      std::filesystem::path save_dir{meshesPath_};
      if(!std::filesystem::exists(save_dir)){
        std::filesystem::create_directory(save_dir);
      }
      const std::string meshFilename = meshesPath_ + "/mesh_kf" + mesh_numbering + ".ply";
      LOG(INFO) << "Saving to path " + meshFilename << std::endl;

      auto start = std::chrono::high_resolution_clock::now();
      okvis::kinematics::Transformation submapPose(seSubmapLookup_[kfId].T_WK.matrix().cast<double>());

      SupereightMapType::SurfaceMesh mesh;
      bool has_cached_mesh = false;
      {
        std::lock_guard<std::mutex> object_lock(submapMeshMutex_);
        const auto mesh_it = seMeshLookup_.find(kfId);
        if (mesh_it != seMeshLookup_.end()) {
          mesh = mesh_it->second;
          has_cached_mesh = true;
        }
      }
      if (!has_cached_mesh) {
        mesh = seSubmapLookup_[kfId].map->mesh();
      }

      se::io::save_mesh(mesh, meshFilename, Eigen::Isometry3f(submapPose.T().cast<float>()));   

      auto end_first = std::chrono::high_resolution_clock::now();

      DLOG(INFO) << "It took " << std::to_string(std::chrono::duration_cast<std::chrono::seconds>(end_first - start).count()) << " seconds to process the mesh";

    }


    void SubmappingInterface::obtainCovisibles(std::vector<uint64_t>& orderedIdx, const okvis::StateId& latestCovisible, int recursionDepth){
      std::set<uint64_t> covisibleMaps;
      std::queue<std::pair<int, okvis::StateId>> covisible_queue;
      std::set<StateId> checkedStates;

      if(seSubmapLookup_.find(latestCovisible.value()) != seSubmapLookup_.end()){
          covisibleMaps.insert(latestCovisible.value());
        }

      for(const auto& state: covisibleFrames_[latestCovisible]) {
        covisible_queue.push({0, state});
        if(seSubmapLookup_.find(state.value()) != seSubmapLookup_.end()){
          covisibleMaps.insert(state.value());
        }
      }

      checkedStates.insert(latestCovisible);

      while(!covisible_queue.empty()) {
        const auto& state = covisible_queue.front();
        if(checkedStates.find(state.second)== checkedStates.end() &&
           covisibleFrames_.find(state.second) != covisibleFrames_.end() &&
           state.first <= recursionDepth){
          checkedStates.insert(state.second);
          for(const auto& covisible: covisibleFrames_[state.second]) {
            covisible_queue.push({state.first + 1, covisible});
            if(seSubmapLookup_.find(covisible.value()) != seSubmapLookup_.end()){
              covisibleMaps.insert(covisible.value());
            }
          }
        }
        covisible_queue.pop();
      }

      for(const auto& covMap: covisibleMaps) {
        orderedIdx.push_back(covMap);
      }
    }

    void SubmappingInterface::processSupereightFrames() {

      SupereightFrames supereightFrame;
      if (gsMappingEnabled_.load()) {
        try {
          if (submapConfig_.gsOptimParamsPath.empty()) {
            throw std::runtime_error("GS runtime enabled but gs_optim_params_path is empty.");
          }
          auto runtime = std::make_shared<GsRuntime>();
          runtime->optimParams =
              gs::param::read_optim_params_from_json(submapConfig_.gsOptimParamsPath);
          runtime->outputDirectory = submapConfig_.gsOutputDirectory.empty()
                                         ? (std::filesystem::path(meshesPath_) / "gs")
                                         : std::filesystem::path(
                                               submapConfig_.gsOutputDirectory);
          std::filesystem::create_directories(runtime->outputDirectory);
          LOG(INFO) << "GS runtime initialized. Params: " << submapConfig_.gsOptimParamsPath
                    << " output: " << runtime->outputDirectory;

          std::atomic_store_explicit(
              &gsRuntime_, runtime, std::memory_order_release);
          if (submapConfig_.gsShowPreview) {
            stopGsPreviewWorker();
            gsPreviewRuntime_ = runtime;
            gsPreviewRuntime_->dataQueue.resume();
            gsPreviewWorker_ = std::thread(
                &SubmappingInterface::runGsPreviewWorker, this, gsPreviewRuntime_);
          }
        } catch (const std::exception& e) {
          LOG(ERROR) << "Failed to initialize GS runtime: " << e.what()
                     << ". Falling back to occupancy-only integration.";
          std::atomic_store_explicit(
              &gsRuntime_, std::shared_ptr<GsRuntime>{},
              std::memory_order_release);
          gsMappingEnabled_.store(false);
        }
      }
      const bool latest_wins =
          submapConfig_.realtimePipelineEnabled && !blocking_;
      while(!isFinished()){
        while (supereightFrames_.PopBlocking(&supereightFrame)){
          isProcessingSeFrame_ = true;
          if (latest_wins) {
            SupereightFrames newer_frame;
            size_t drained_frames = 0;
            while (supereightFrames_.PopNonBlocking(&newer_frame)) {
              supereightFrame = std::move(newer_frame);
              ++drained_frames;
            }
            recordLatestWinsDrops("Supereight frames before integration",
                                  realtimeSupereightDrainDropCount_, drained_frames);

            if (supereightFrame.vecDepthFrames.size() > 1) {
              const size_t dropped_depth_frames = supereightFrame.vecDepthFrames.size() - 1;
              auto latest_depth_frame = std::move(supereightFrame.vecDepthFrames.back());
              supereightFrame.vecDepthFrames.clear();
              supereightFrame.vecDepthFrames.push_back(std::move(latest_depth_frame));
              recordLatestWinsDrops("RGB-D packets packed in a Supereight frame",
                                    realtimePackedDepthDropCount_, dropped_depth_frames);
            }
          }

          TimerSwitchable timingProcessSeFrame("9 Process supereight frames");

          TimerSwitchable timingNewSubmap("9.1 Decide a new submap");
          bool create_new_submap = decideNewSubmap(supereightFrame);
          timingNewSubmap.stop();


          auto ensureGsSubmapState =
              [&](const uint64_t keyframe_id) -> GsRuntime::GsSubmapState* {
            if (!gsRuntime_) {
              return nullptr;
            }
            auto& gs_state = gsRuntime_->submapStates[keyframe_id];
            const std::filesystem::path submap_output_dir =
                gsRuntime_->outputDirectory / ("submap_" + std::to_string(keyframe_id));
            if (gs_state.creationIndex == 0) {
              gs_state.creationIndex = ++gsRuntime_->nextSubmapCreationIndex;
            }
            if (gs_state.outputDirectory.empty()) {
              gs_state.outputDirectory = submap_output_dir;
            }
            if (!gs_state.model) {
              if (gs_state.offloadQueued) {
                waitForGsModelOffloads();
              }
              std::filesystem::create_directories(submap_output_dir);
              gs_state.model = std::make_unique<gs::GaussianModel>(
                  gsRuntime_->optimParams, submap_output_dir.string());
              gs_state.model->densify_params =
                  toGsErrorDensifyParams(gsDensifyConfig_);
              if (gs_state.hasCpuModelSnapshot) {
                gs_state.model->Import_ply_tensors_cpu(gs_state.cpuModelSnapshot);
              }
            }
            return &gs_state;
          };

          for (auto &frameData: supereightFrame.frameData_) {

            const Transformation T_WS = frameData.second.T_WS;
            if(covisibleFrames_.find(StateId(frameData.first)) == covisibleFrames_.end()){
              covisibleFrames_.emplace(StateId(frameData.first), frameData.second.covisibles_);
            } else {
              covisibleFrames_[StateId(frameData.first)] = frameData.second.covisibles_;
            }

            if (seSubmapLookup_.count(frameData.first)) {

              seSubmapLookup_[frameData.first].T_WK = Eigen::Isometry3f(T_WS.T().cast<float>());

            } else if(frameData.first == supereightFrame.keyFrameId){

              seSubmapLookup_[supereightFrame.keyFrameId] = {nullptr, Eigen::Isometry3f(T_WS.T().cast<float>())};

            }
          }

          if (create_new_submap) {

            updated_octants_.clear();

            submapCounter_++;
            gsPreviewSubmapCount_.store(submapCounter_, std::memory_order_relaxed);
            if(submapCounter_ > 1){
              previousSubmap_ = seSubmapLookup_[prevKeyframeId_].map.get();
              previousSubmapId_ = prevKeyframeId_;
            }

            addSubmapAlignmentFactors(supereightFrame);

            LOG(INFO) << "New submap generation with counter " << submapCounter_ << " (kf Id: " << supereightFrame.keyFrameId << ")";

            seSubmapLookup_[supereightFrame.keyFrameId].map = std::shared_ptr<SupereightMapType>(new SupereightMapType(mapConfig_, dataConfig_));
            markOccupancySubmapDirty(supereightFrame.keyFrameId);
            if (gsMappingEnabled_.load() && gsRuntime_) {
              ensureGsSubmapState(supereightFrame.keyFrameId);
            }
            prevKeyframeId_ = supereightFrame.keyFrameId;
            if (gsFisherVisualizationEnabled_.load()) {
              clearInactiveFisherOverlays(prevKeyframeId_);
              gsFisherVisualizationRefreshRequested_.store(true);
            }
            if (gsMappingEnabled_.load() && gsRuntime_) {

              updateGsSubmapResidencyPolicy(prevKeyframeId_);
            }

            lidarKfDetected_ = false;
            numIntegratedLidarFrames_ = 0;
            numIntegratedDepthFrames_ = 0; 
          }
          auto& activeMap = *(seSubmapLookup_[supereightFrame.keyFrameId].map);

          se::MapIntegrator integrator(activeMap);

          TimerSwitchable batchIntegration("9.2 Lidar batch integration");

          batchIntegration.stop();

          if(cameraSensors_ && (latest_wins || supereightFrames_.Size() <= 1)){
            okvis::kinematics::Transformation T_WK(seSubmapLookup_[prevKeyframeId_].T_WK.matrix().cast<double>());

            for(size_t i = 0; i < supereightFrame.vecDepthFrames.size(); i++){
              TimerSwitchable diIntegration("9.3 Depth integration");
              try{
                int depthImage_idx = -1;
                int colourImage_idx = -1;
                for(const auto& idx_cam_measurements: supereightFrame.vecDepthFrames[i]) {
                  if((viParameters_.nCameraSystem.isDepthCamera(idx_cam_measurements.first)
                      || viParameters_.mappingInputEnabledFor(idx_cam_measurements.first)) &&
                    (*cameraSensors_).find(idx_cam_measurements.first) != (*cameraSensors_).end()){
                      for(const auto& measurement : idx_cam_measurements.second) {
                        if(!measurement.second.measurement.depthImage.empty()) {
                          depthImage_idx = idx_cam_measurements.first;
                        }
                      }
                  }

                  if((viParameters_.nCameraSystem.cameraType(idx_cam_measurements.first).isColour
                      || viParameters_.mappingInputEnabledFor(idx_cam_measurements.first))
                            && (*cameraSensors_).find(idx_cam_measurements.first) != (*cameraSensors_).end()) {
                      for(const auto& measurement : idx_cam_measurements.second) {
                        if(!measurement.second.measurement.image.empty()) {
                          colourImage_idx = idx_cam_measurements.first;
                        }
                      }
                  }
                }

                std::pair<Eigen::Isometry3f, okvis::CameraMeasurement> depthData;
                std::pair<Eigen::Isometry3f, okvis::CameraMeasurement> rgbData;

                auto integrateDepthMeasurements = [&](const auto& se_measurements,
                                                     const bool has_rgb_for_gs) {
                  bool integrated_with_gs = false;
                  bool gs_occupancy_completed = false;
                  std::shared_ptr<RgbFrameSourceLease> rgb_source_lease;
                  std::unordered_set<const se::OctantBase*> updated_octants_local;
                  std::unordered_set<const se::OctantBase*>* updated_octants_ptr =
                      &updated_octants_local;
                  const bool try_gs = gsMappingEnabled_.load() && has_rgb_for_gs && gsRuntime_;
                  if (try_gs) {
                    auto* gs_state = ensureGsSubmapState(supereightFrame.keyFrameId);
                    if (gs_state && gs_state->model) {
                      const size_t gs_cam_cpu_count_before = gs_state->gsCamListCpu.size();
                      gs::PoseUpdateVector gs_pose_updates;
                      gs::FisherTrainingSampleVector gs_fisher_training_samples;
                      gs::DataPacket gs_preview_packet;

                      gs::DataPacket* gs_preview_packet_out =
                          submapConfig_.gsShowPreview ? &gs_preview_packet : nullptr;

                      try {
                        auto gs_measurements = se_measurements;
                        const cv::Mat& rgb_mat = rgbData.second.measurement.image;
                        gs::RgbInputView rgb_input;
                        rgb_input.host_ptr = rgb_mat.ptr<uint8_t>();
                        rgb_input.width = rgb_mat.cols;
                        rgb_input.height = rgb_mat.rows;
                        rgb_input.row_stride = rgb_mat.step[0];
                        rgb_input.channel_order = gs::RgbChannelOrder::RGB;

                        auto rgb_lifetime =
                            rgbData.second.measurement.imageBufferLifetime;
                        if (rgb_lifetime) {
                          const auto* owner_host =
                              reinterpret_cast<const uint8_t*>(
                                  rgb_lifetime->hostData());
                          const auto* owner_mapped =
                              reinterpret_cast<const uint8_t*>(
                                  rgb_lifetime->mappedDeviceData());
                          if (owner_host && owner_mapped) {
                            const uintptr_t input_address =
                                reinterpret_cast<uintptr_t>(rgb_input.host_ptr);
                            const uintptr_t owner_address =
                                reinterpret_cast<uintptr_t>(owner_host);
                            if (input_address >= owner_address) {
                              const size_t offset = static_cast<size_t>(
                                  input_address - owner_address);
                              const size_t required_bytes =
                                  rgb_input.row_stride
                                  * static_cast<size_t>(rgb_input.height);
                              const size_t owner_capacity =
                                  rgb_lifetime->capacityBytes();
                              if (offset <= owner_capacity
                                  && required_bytes
                                         <= owner_capacity - offset) {
                                rgb_input.mapped_device_ptr =
                                    owner_mapped + offset;
                              }
                            }
                          }
                          switch (rgb_lifetime->memoryKind()) {
                            case FrameBufferMemoryKind::CudaRegisteredMapped:
                              rgb_input.memory_kind =
                                  gs::RgbInputMemoryKind::RegisteredMapped;
                              break;
                            case FrameBufferMemoryKind::CudaPinnedStaging:
                              rgb_input.memory_kind =
                                  gs::RgbInputMemoryKind::PinnedStaging;
                              break;
                            case FrameBufferMemoryKind::OrdinaryHost:
                            default:
                              rgb_input.memory_kind =
                                  gs::RgbInputMemoryKind::OrdinaryHost;
                              break;
                          }
                          rgb_source_lease =
                              std::make_shared<RgbFrameSourceLease>(
                                  std::move(rgb_lifetime));
                          rgb_input.owner = rgb_source_lease;
                          rgb_input.record_cuda_use =
                              [rgb_source_lease](void* native_stream) noexcept {
                                rgb_source_lease->recordCudaUse(native_stream);
                              };
                          const uint8_t* source_rgb_data = rgb_mat.data;
                          rgb_input.release_owner =
                              [rgb_source_lease,
                               &rgbData,
                               &supereightFrame,
                               i,
                               colourImage_idx,
                               source_rgb_data]() noexcept {
                            rgbData.second.measurement.image.release();
                            auto source_map_it =
                                supereightFrame.vecDepthFrames[i].find(
                                    colourImage_idx);
                            if (source_map_it
                                != supereightFrame.vecDepthFrames[i].end()) {
                              for (auto& source_measurement :
                                   source_map_it->second) {
                                if (source_measurement.second.measurement.image.data
                                    == source_rgb_data) {
                                  source_measurement.second.measurement.image.release();
                                }
                              }
                            }
                            rgb_source_lease->release();
                          };

                          rgbData.second.measurement.imageBufferLifetime.reset();
                          auto& source_measurements =
                              supereightFrame.vecDepthFrames[i].at(colourImage_idx);
                          for (auto& source_measurement : source_measurements) {
                            if (source_measurement.second.measurement.image.data
                                == rgb_mat.data) {
                              source_measurement.second.measurement
                                  .imageBufferLifetime.reset();
                            }
                          }
                        }
                        gs_measurements.rgb_input = std::move(rgb_input);

                        integrator.integrateDepth(integration_counter_,
                                                  gs_measurements,
                                                  *(gs_state->model),
                                                  gs_state->gsCamList,
                                                  gs_state->gtImgList,
                                                  gs_state->geometryTargetList,
                                                  gs_state->depthTargetListCpu,
                                                  gsRuntime_->dataQueue,
                                                  updated_octants_ptr,
                                                  &gs_state->gsCamListCpu,
                                                  &gs_state->gtImgListCpu,
                                                  &gs_pose_updates,
                                                  &gs_fisher_training_samples,
                                                  gs_preview_packet_out,
                                                  &gs_occupancy_completed);
                        integrated_with_gs = true;
                        const size_t gs_cam_cpu_count_after = gs_state->gsCamListCpu.size();

                        if (colourImage_idx >= 0 && has_rgb_for_gs) {
                          kinematics::Transformation T_SC_meta =
                              *viParameters_.mappingT_SC(colourImage_idx);
                          {
                            std::lock_guard<std::mutex> state_lock(state_mtx_);
                            if (latestState_.isOnlineExtrinsics) {
                              const auto ext_it =
                                  latestState_.extrinsics.find(colourImage_idx);
                              if (ext_it != latestState_.extrinsics.end()) {
                                T_SC_meta = ext_it->second;
                              }
                            }
                          }

                          if (gs_cam_cpu_count_after < gs_cam_cpu_count_before) {
                            LOG(WARNING) << "GS pose metadata append skipped due to camera list "
                                         << "size regression (before="
                                         << gs_cam_cpu_count_before << ", after="
                                         << gs_cam_cpu_count_after << ")";
                            invalidateGsSubmapFisherCache(supereightFrame.keyFrameId);
                          } else {
                            const Eigen::Matrix4f T_SC_f = T_SC_meta.T().cast<float>();
                            const uint64_t timestamp_ns =
                                static_cast<uint64_t>(rgbData.second.timeStamp.toNSec());
                            uint64_t source_state_id = 0;
                            {
                              std::lock_guard<std::mutex> lock(gsPoseStateIdByTimestampMtx_);
                              const auto it_state =
                                  gsPoseStateIdByTimestamp_.find(timestamp_ns);
                              if (it_state != gsPoseStateIdByTimestamp_.end()) {
                                source_state_id = it_state->second;
                                gsPoseStateIdByTimestamp_.erase(it_state);
                              }
                            }
                            gs_state->poseMetadataCpu.reserve(
                                gs_state->poseMetadataCpu.size()
                                + (gs_cam_cpu_count_after - gs_cam_cpu_count_before));
                            for (size_t sample_idx = gs_cam_cpu_count_before;
                                 sample_idx < gs_cam_cpu_count_after; ++sample_idx) {
                              GsRuntime::GsSamplePoseMetadata pose_meta;
                              pose_meta.timestamp_ns = timestamp_ns;
                              pose_meta.state_id = source_state_id;
                              pose_meta.cam_idx = colourImage_idx;
                              pose_meta.T_SC = T_SC_f;
                              gs_state->poseMetadataCpu.push_back(pose_meta);
                            }

                            if (!gs_pose_updates.empty()) {
                              std::map<uint64_t, Eigen::Matrix4f> pose_updates_by_state;
                              std::map<uint64_t, Eigen::Vector3f> traj_updates_by_state;
                              const size_t meta_size = gs_state->poseMetadataCpu.size();
                              const Eigen::Isometry3f T_WKf(T_WK.T().cast<float>());
                              const Eigen::Isometry3f T_SC_current(T_SC_f);
                              for (const auto& pose_update : gs_pose_updates) {
                                uint64_t target_state_id = 0;
                                Eigen::Isometry3f T_SC_for_state = T_SC_current;

                                if (pose_update.sample_index
                                    != std::numeric_limits<size_t>::max()
                                    && pose_update.sample_index < meta_size) {
                                  const auto& pose_meta =
                                      gs_state->poseMetadataCpu[pose_update.sample_index];
                                  target_state_id = pose_meta.state_id;
                                  T_SC_for_state = Eigen::Isometry3f(pose_meta.T_SC);
                                }
                                if (target_state_id == 0 && pose_update.is_current_frame) {
                                  target_state_id = source_state_id;
                                }
                                if (target_state_id == 0) {
                                  continue;
                                }

                                Eigen::Isometry3f T_KC = Eigen::Isometry3f::Identity();
                                T_KC.matrix() = pose_update.T_WC;
                                const Eigen::Isometry3f T_WC = T_WKf * T_KC;
                                const Eigen::Isometry3f T_WS =
                                    T_WC * T_SC_for_state.inverse();
                                pose_updates_by_state[target_state_id] = T_WS.matrix();
                                traj_updates_by_state[target_state_id] = T_WS.translation();
                              }

                              if (!pose_updates_by_state.empty()) {
                                gsOptimizedPoseCallback gs_optimized_pose_callback;
                                {
                                  std::lock_guard<std::mutex> lock(
                                      gsOptimizedPoseCallbackMtx_);
                                  gs_optimized_pose_callback = gsOptimizedPoseCallback_;
                                }
                                if (gs_optimized_pose_callback) {
                                  gs_optimized_pose_callback(pose_updates_by_state);
                                }

                                gsOptimizedTrajectoryCallback gs_optimized_traj_callback;
                                {
                                  std::lock_guard<std::mutex> lock(
                                      gsOptimizedTrajectoryCallbackMtx_);
                                  gs_optimized_traj_callback =
                                      gsOptimizedTrajectoryCallback_;
                                }
                                if (gs_optimized_traj_callback
                                    && !traj_updates_by_state.empty()) {
                                  gs_optimized_traj_callback(traj_updates_by_state);
                                }
                              }
                            }
                          }
                        }

                        if (!gs_fisher_training_samples.empty()) {
                          updateGsSubmapFisherFromTrainingSamples(
                              supereightFrame.keyFrameId, gs_fisher_training_samples);
                        }
                        if (gs_preview_packet_out && gs_preview_packet.ID >= 0) {

                          gsRuntime_->dataQueue.push(std::move(gs_preview_packet));
                        }

                        gsIntegrationFailureCount_ = 0;

                      } catch (const std::exception& e) {
                        const bool fatal_gs_error = isFatalGsIntegrationFailure(e);
                        if (fatal_gs_error) {
                          ++gsIntegrationFailureCount_;
                        } else {
                          gsIntegrationFailureCount_ = 0;
                        }

                        if (gsIntegrationFailureCount_ <= 3) {
                          LOG(WARNING) << "GS integration failed. Falling back to occupancy-only "
                                          "for this frame. Reason: "
                                       << e.what();
                        }
                        if (fatal_gs_error && !gsRuntimeDisabledDueToErrors_
                            && gsIntegrationFailureCount_ >= 3) {
                          gsRuntimeDisabledDueToErrors_ = true;
                          LOG(ERROR) << "Disabling GS runtime after "
                                     << gsIntegrationFailureCount_
                                     << " consecutive integration failures to keep "
                                        "online pipeline stable.";
                          stopGsPreviewWorker();
                          waitForGsModelOffloads();
                          std::atomic_store_explicit(
                              &gsRuntime_, std::shared_ptr<GsRuntime>{},
                              std::memory_order_release);
                          gsMappingEnabled_.store(false);
                        }
                      }
                    }
                  }
                  if (!integrated_with_gs && !gs_occupancy_completed) {
                    integrator.integrateDepth(
                        integration_counter_, se_measurements, updated_octants_ptr);
                  }

                  if (!updated_octants_local.empty()) {
                    markOccupancySubmapDirty(
                        supereightFrame.keyFrameId, &updated_octants_local);
                  }
                };

                if(depthImage_idx == -1) {
                  LOG(WARNING) << "No integration is going to happen, either there is no depth camera with the mapping flag set to true or there are no depth images";
                  continue;
                } else if(depthImage_idx != -1 && colourImage_idx == -1) {
                  DLOG(INFO) << "Only depth";
                  bool found_depth = false;

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(depthImage_idx)) {
                    if(!measurements.second.measurement.depthImage.empty()) {
                      depthData = measurements;
                      found_depth = true;
                      break;
                    }
                  }

                  if (!found_depth) {
                    LOG(WARNING) << "Skipping depth-only integration: selected depth camera "
                                 << depthImage_idx << " has no valid depth image in this frame.";
                    continue;
                  }

                  se::Measurements seMeasurements = {se::Measurement{
                    depthMat2ImageView(depthData.second.measurement.depthImage),
                    (*cameraSensors_).at(depthImage_idx).second,
                    Eigen::Isometry3f(T_WK.inverse().T().cast<float>() * depthData.first.matrix())}};
                  std::unordered_set<const se::OctantBase*> updated_octants_local;
                  integrator.integrateDepth(
                      integration_counter_, seMeasurements, &updated_octants_local);

                  if (!updated_octants_local.empty()) {
                    markOccupancySubmapDirty(
                        supereightFrame.keyFrameId, &updated_octants_local);
                  }
                  integration_counter_++;
                  numIntegratedDepthFrames_++;
                } else if(depthImage_idx != -1 && colourImage_idx != -1) {
                  bool found_depth = false;
                  bool found_rgb = false;

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(depthImage_idx)) {
                    if(!measurements.second.measurement.depthImage.empty()) {
                      depthData = measurements;
                      found_depth = true;
                      break;
                    }
                  }

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(colourImage_idx)) {
                    if(!measurements.second.measurement.image.empty()) {
                      rgbData = measurements;
                      found_rgb = true;
                      break;
                    }
                  }

                  if (!found_depth) {
                    LOG(WARNING) << "Skipping RGB-D integration: depth camera "
                                 << depthImage_idx
                                 << " has no valid depth image in this frame.";
                    continue;
                  }
                  if (!found_rgb) {
                    LOG(WARNING) << "Skipping RGB-D integration: color camera "
                                 << colourImage_idx
                                 << " has no valid RGB image in this frame.";
                    continue;
                  }

                  const auto& depthSensor = (*cameraSensors_).at(depthImage_idx).second;
                  const auto& colorSensor = (*cameraSensors_).at(colourImage_idx).second;
                  const Eigen::Matrix4f T_KWf = T_WK.inverse().T().cast<float>();
                  const Eigen::Isometry3f T_KD(T_KWf * depthData.first.matrix());
                  const Eigen::Isometry3f T_KC(T_KWf * rgbData.first.matrix());

                  {
                    integrateDepthMeasurements(
                        se::Measurements{se::Measurement{
                                            depthMat2ImageView(depthData.second.measurement.depthImage),
                                            depthSensor,
                                            T_KD},
                                          se::Measurement<se::PinholeCamera, se::colour_t>{
                                            rgbMat2ImageView(rgbData.second.measurement.image),
                                            colorSensor,
                                            T_KC}
                                          },
                        true);
                  }
                  integration_counter_++;
                  numIntegratedDepthFrames_++;
                }
                diIntegration.stop();

                if (submapConfig_.useMap2MapFactors) {
                  updateDepthAlignBlock(depthData, T_WK, depthImage_idx);
                }
              } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
              }   
            } 
          }

          okvis::State current_state; 
          Eigen::Matrix4f currentPose_WS;
          {
            std::lock_guard lockCurrentPose(currentPose_mtx_);
            currentPose_WS = currentPose_;
          }
          current_state.T_WS = okvis::kinematics::Transformation(currentPose_WS.cast<double>());

          if (occupancyViewerConfig_.enable) {
            updateOccupancyPreview(current_state);
          }
          updateGsFisherVisualization();

          if(create_new_submap && previousSubmapId_ != UNINITIALIZED_ID){

            LOG(INFO) << "Completed integrating submap " << previousSubmapId_ << " which is submap number " << seSubmapLookup_.size();
            LOG(INFO) << okvis::timing::Timing::print();

            DLOG(INFO) << "Trying to mesh for frame " << previousSubmapId_;
            if(previousSubmapId_ != UNINITIALIZED_ID){
              enqueueSubmapFinalization(previousSubmapId_);
            }
          }

          isProcessingSeFrame_ = false;
          timingProcessSeFrame.stop();
        }
      }
      return;
    }


    DepthFrame SubmappingInterface::depthMat2ImageView(const cv::Mat &inputDepth) {
      if (inputDepth.type() != CV_32FC1) {
        throw std::runtime_error("Only implemented for CV_32FC1 cv::Mat");
      }
      if (!inputDepth.isContinuous()) {
        throw std::runtime_error("Only implemented for continuous cv::Mat");
      }
      return DepthFrame(inputDepth.cols, inputDepth.rows, const_cast<float*>(inputDepth.ptr<float>()));
    }


    se::Image<se::colour_t> SubmappingInterface::rgbMat2ImageView(const cv::Mat &inputRGB) {
      if (inputRGB.type() != CV_8UC3) {
        throw std::runtime_error("Only implemented for CV_8UC3 cv::Mat");
      }
      if (!inputRGB.isContinuous()) {
        throw std::runtime_error("Only implemented for continuous cv::Mat");
      }
      return se::Image<se::colour_t>(inputRGB.cols,
                                     inputRGB.rows,
                                     reinterpret_cast<se::colour_t*>(inputRGB.data));
    }

    bool SubmappingInterface::checkForAvailableData() {
      DepthMeasurementPtr depthMeasurement;
      LidarMeasurement oldestLidarMeasurement;
      bool integrateLidar = true;
      bool integrateDepthImage = true;

      if (!lidarMeasurements_.getCopyOfFront(&oldestLidarMeasurement) && !depthMeasurements_.getCopyOfFront(&depthMeasurement)){
        integrateLidar = false;
        integrateDepthImage = false;
      }

      OkvisUpdate newestState;
      if(!stateUpdates_.getCopyOfBack(&newestState)){
        DLOG(INFO) << "not having any state updates";
        return false;
      }

      if(!(oldestLidarMeasurement.timeStamp <= newestState.timestamp) || !lidarSensors_){
        integrateLidar = false;
      }

      okvis::Time newestDataTimestamp;
      if(!cameraSensors_){
        newestDataTimestamp = oldestLidarMeasurement.timeStamp;
        integrateDepthImage = false;
      } else if(depthMeasurement && !depthMeasurement->empty()){
        for(const auto& idx_cam_data : *depthMeasurement) {
          for(const auto& measurement : idx_cam_data.second) {
            if(measurement.timeStamp > newestDataTimestamp) {
              newestDataTimestamp = measurement.timeStamp;
            }
          }
        }

        if(newestDataTimestamp > newestState.timestamp) {
            integrateDepthImage = false;
        }
      }

      if(!(integrateDepthImage || integrateLidar)){
        OkvisUpdate popState;
        bool continuePopping = true;
        while(continuePopping){

          if(blocking_) {
            continuePopping = stateUpdates_.PopBlocking(&popState);
          } else {
            continuePopping = stateUpdates_.PopNonBlocking(&popState);
          }

          while(trajectoryLocked_);
          trajectoryLocked_ = true;

          auto updated_mut = std::make_shared<okvis::AlignedMap<okvis::StateId, okvis::State>>(
              *(popState.updatedStates));
          (*updated_mut)[popState.latestState.id] = popState.latestState;
          std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated_const(updated_mut);

          std::set<okvis::StateId> affected = popState.affectedStates;
          affected.insert(popState.latestState.id);

          propagatedStates.update(popState.trackingState, updated_const, affected);
          trajectoryLocked_ = false;

          frameUpdater(popState);

          if(continuePopping) {
            continuePopping = popState.timestamp < newestState.timestamp;
          }

        }

        return false;
      } 

      OkvisUpdate currentStateUpdate;
      bool continue_popping = true;

      while(continue_popping) {
        if(blocking_) {
          stateUpdates_.PopBlocking(&currentStateUpdate);
        }
        else {
          stateUpdates_.PopNonBlocking(&currentStateUpdate);
        }

        while(trajectoryLocked_);
        trajectoryLocked_ = true;

        auto updated_mut = std::make_shared<okvis::AlignedMap<okvis::StateId, okvis::State>>(
            *(currentStateUpdate.updatedStates));
        (*updated_mut)[currentStateUpdate.latestState.id] = currentStateUpdate.latestState;
        std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated_const(updated_mut);

        std::set<okvis::StateId> affected = currentStateUpdate.affectedStates;
        affected.insert(currentStateUpdate.latestState.id);

        propagatedStates.update(currentStateUpdate.trackingState, updated_const, affected);
        trajectoryLocked_ = false;

        if(!frame_) {
          frameUpdater(currentStateUpdate);
        }

        if(continue_popping) {
            continue_popping = currentStateUpdate.timestamp < newestDataTimestamp;
        }
      }

      bool keepLooping = true;
      bool processedRealtimeDepth = false;
      Transformation T_WD;

      LidarMeasurement lidarMeasurement;
      lidarMeasurements_.getCopyOfFront(&lidarMeasurement);
      depthMeasurements_.getCopyOfFront(&depthMeasurement);

      std::vector<okvis::State> states;
      std::vector<okvis::Time> times;
      std::vector<okvis::LidarMeasurement, Eigen::aligned_allocator<okvis::LidarMeasurement>> measurements;

      while(keepLooping){

        if(integrateLidar && lidarMeasurement.timeStamp <= currentStateUpdate.timestamp){

          if(blocking_) {
            lidarMeasurements_.PopBlocking(&lidarMeasurement);
          }
          else {
            lidarMeasurements_.PopNonBlocking(&lidarMeasurement);
          }

          measurements.push_back(lidarMeasurement);
          times.push_back(lidarMeasurement.timeStamp);
        }
        bool depthImageExisting = integrateDepthImage
                                  && (blocking_ || !processedRealtimeDepth);
        size_t mappingCameraId = 0;
        if(cameraSensors_) {
          mappingCameraId = (*cameraSensors_).begin()->first;
        }
        if(depthImageExisting && depthMeasurement && !depthMeasurement->empty()){
          const auto mapping_cam_it = depthMeasurement->find(mappingCameraId);
          if (mapping_cam_it == depthMeasurement->end() || mapping_cam_it->second.empty()) {
            depthImageExisting = false;
          } else {
            depthImageExisting =
                depthImageExisting
                && (mapping_cam_it->second.front().timeStamp <= currentStateUpdate.timestamp);
          }
        } else {
          depthImageExisting = false;
        }
        if(depthImageExisting){

          if(blocking_) {
            depthMeasurements_.PopBlocking(&depthMeasurement);
          }
          else {
            depthMeasurements_.PopNonBlocking(&depthMeasurement);

            size_t drained_depth_frames = 0;
            DepthMeasurementPtr newer_depth_measurement;
            while (depthMeasurements_.getCopyOfFront(&newer_depth_measurement)
                   && depthMeasurementReadyForState(newer_depth_measurement,
                                                    mappingCameraId,
                                                    currentStateUpdate.timestamp)) {
              if (!depthMeasurements_.PopNonBlocking(&newer_depth_measurement)) {
                break;
              }
              depthMeasurement = std::move(newer_depth_measurement);
              ++drained_depth_frames;
            }
            recordLatestWinsDrops("RGB-D packets before pose lookup",
                                  realtimeDepthDrainDropCount_, drained_depth_frames);
            processedRealtimeDepth = true;
          }

          okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>> posedCameraMeasurements;
          bool all_assigned = true;
          for(auto& cam_idx_measurements : *depthMeasurement) {
            std::vector<std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>> posed_cam_idx_measurements;
            okvis::kinematics::Transformation T_WD;
            for(auto& measurement : cam_idx_measurements.second){
              kinematics::Transformation T_SCi =
                  *viParameters_.mappingT_SC(cam_idx_measurements.first);
              if (currentStateUpdate.latestState.isOnlineExtrinsics) {
                T_SCi = currentStateUpdate.latestState.extrinsics[cam_idx_measurements.first];
              }

              StateId source_state_id;
              if(!predict(measurement.timeStamp, T_SCi, T_WD, &source_state_id)){
                LOG(WARNING) << "Depth image skipped (pose unavailable) t=" << measurement.timeStamp
                            << " cam=" << cam_idx_measurements.first;
                continue;
              }
              if (gsMappingEnabled_.load() && source_state_id.isInitialised()
                  && (viParameters_.nCameraSystem.cameraType(
                          cam_idx_measurements.first).isColour
                      || viParameters_.mappingInputEnabledFor(
                          cam_idx_measurements.first))) {
                const uint64_t timestamp_ns =
                    static_cast<uint64_t>(measurement.timeStamp.toNSec());
                std::lock_guard<std::mutex> lock(gsPoseStateIdByTimestampMtx_);
                gsPoseStateIdByTimestamp_[timestamp_ns] = source_state_id.value();
                constexpr size_t kMaxGsPoseStateLookupSize = 20000;
                constexpr size_t kTrimmedGsPoseStateLookupSize = 15000;
                if (gsPoseStateIdByTimestamp_.size() > kMaxGsPoseStateLookupSize) {
                  auto erase_it = gsPoseStateIdByTimestamp_.begin();
                  const size_t to_erase =
                      gsPoseStateIdByTimestamp_.size() - kTrimmedGsPoseStateLookupSize;
                  for (size_t n = 0; n < to_erase
                                     && erase_it != gsPoseStateIdByTimestamp_.end();
                       ++n) {
                    erase_it = gsPoseStateIdByTimestamp_.erase(erase_it);
                  }
                }
              }

              const bool mapping_input_preprocessed =
                  viParameters_.mappingInputEnabledFor(cam_idx_measurements.first);
              if (mapping_input_preprocessed) {
                if (!viParameters_.mappingInputPreprocessor->process(
                        cam_idx_measurements.first,
                        measurement,
                        static_cast<size_t>(depthImageResDownsamplingRatio_))) {
                  throw std::runtime_error(
                      "mapping input target camera does not match incoming measurement");
                }
              }

              if((viParameters_.nCameraSystem.isDepthCamera(cam_idx_measurements.first)
                  || mapping_input_preprocessed)
                 && !measurement.measurement.depthImage.empty()){

                if (depthScaleFactor_ != 1.0f) {
                  measurement.measurement.depthImage =
                      depthScaleFactor_ * measurement.measurement.depthImage;

                  measurement.measurement.depthBufferLifetime.reset();
                }
                if(!mapping_input_preprocessed && depthImageResDownsamplingRatio_ > 1 ) {
                  if (!cameraSensors_) {
                    throw std::runtime_error("Depth downsampling requires a mapping camera sensor");
                  }
                  const auto sensor_it = cameraSensors_->find(cam_idx_measurements.first);
                  if (sensor_it == cameraSensors_->end()) {
                    throw std::runtime_error("Missing mapping camera sensor for depth downsampling");
                  }
                  const auto& sensor = sensor_it->second.second;
                  const cv::Size target_size(sensor.model.imageWidth(),
                                             sensor.model.imageHeight());
                  static thread_local SharedFrameSlotRing resized_depth_slots(3);
                  auto depth_slot = resized_depth_slots.acquire(
                      target_size.height, target_size.width, CV_32FC1);
                  cv::Mat depthImageSubsampled = depth_slot ? depth_slot->mat() : cv::Mat();
                  downsampleDepth(measurement.measurement.depthImage,
                                          depthImageResDownsamplingRatio_,
                                          target_size,
                                          sensor.near_plane,
                                          sensor.far_plane,
                                          depthImageSubsampled);
                  measurement.measurement.depthImage = std::move(depthImageSubsampled);
                  measurement.measurement.depthBufferLifetime = std::move(depth_slot);
                }
              }

              if (!mapping_input_preprocessed
                  && depthImageResDownsamplingRatio_ > 1
                  && viParameters_.nCameraSystem.cameraType(cam_idx_measurements.first).isColour
                  && !measurement.measurement.image.empty() && cameraSensors_) {
                const auto sensor_it = cameraSensors_->find(cam_idx_measurements.first);
                if (sensor_it != cameraSensors_->end()) {
                  const cv::Size target_size(sensor_it->second.second.model.imageWidth(),
                                             sensor_it->second.second.model.imageHeight());
                  if (target_size.width > 0 && target_size.height > 0
                      && measurement.measurement.image.size() != target_size) {
                    static thread_local SharedFrameSlotRing resized_rgb_slots(3);
                    auto slot = resized_rgb_slots.acquire(
                        target_size.height, target_size.width, CV_8UC3);
                    if (slot) {
                      cv::resize(measurement.measurement.image,
                                 slot->mat(),
                                 target_size,
                                 0.0,
                                 0.0,
                                 cv::INTER_AREA);
                      measurement.measurement.image = slot->mat();
                      measurement.measurement.imageBufferLifetime = std::move(slot);
                    }
                    else {
                      cv::Mat resized_rgb;
                      cv::resize(measurement.measurement.image,
                                 resized_rgb,
                                 target_size,
                                 0.0,
                                 0.0,
                                 cv::INTER_AREA);
                      measurement.measurement.image = std::move(resized_rgb);
                      measurement.measurement.imageBufferLifetime.reset();
                    }
                  }
                }
              }

              posed_cam_idx_measurements.emplace_back(std::make_pair(Eigen::Isometry3f(T_WD.T().cast<float>()), measurement));
            }

            posedCameraMeasurements.emplace(std::make_pair(cam_idx_measurements.first, posed_cam_idx_measurements));
          }

          if(!all_assigned) {
            break;
          }

          if(frame_){
              frame_->vecDepthFrames.push_back(posedCameraMeasurements);
          } else {
            throw std::runtime_error("When adding image data we have found a point where there is no frame initialized where there should have be");
          }
        }

        bool keepLoopingDepth = true;
        if(!blocking_) {

            keepLoopingDepth = false;
        }
        else if(!integrateDepthImage || !depthMeasurements_.getCopyOfFront(&depthMeasurement)){
            keepLoopingDepth = false;
        }
        else {
          const auto mapping_cam_it = depthMeasurement->find(mappingCameraId);
          if (mapping_cam_it == depthMeasurement->end() || mapping_cam_it->second.empty()
              || mapping_cam_it->second.front().timeStamp >= currentStateUpdate.timestamp) {
            keepLoopingDepth = false;
          }
        }
        bool keepLoopingLidar = true;
        if((!lidarMeasurements_.getCopyOfFront(&lidarMeasurement) || lidarMeasurement.timeStamp >= currentStateUpdate.timestamp || !integrateLidar)){
            keepLoopingLidar = false;
        }

        if(!keepLoopingDepth && !keepLoopingLidar ) {
          keepLooping = false;
          if(integrateLidar){
            states.reserve(times.size());

            bool getStatesSuccessfully = propagatedStates.getStates(times, states);
            if(getStatesSuccessfully){
              for(size_t i = 0; i < times.size(); i++){
                Eigen::Matrix4f T_WL = states[i].T_WS.T().cast<float>() * (*viParameters_.lidar).T_SL.T().cast<float>();
                if(!frame_){
                  throw std::runtime_error("About to add lidar measurements and we have found a point where there is no frame initialized where there should have be");
                }else{
                  frame_->vecRayMeasurements.push_back({Eigen::Isometry3f(T_WL), 
                                                        measurements[i].measurement.rayMeasurement.template cast<float>()});
                }
              }
            }
          }

          frameUpdater(currentStateUpdate);

          if(frame_){
            if(blocking_){
              supereightFrames_.PushBlockingIfFull(*frame_, 1);
            } else if(supereightFrames_.PushNonBlockingDroppingIfFull(
                          *frame_, nonBlockingSupereightQueueSize_)){
              recordLatestWinsDrops("Supereight queue entries",
                                    realtimeSupereightQueueDropCount_, 1);
            }
            frame_.reset();
          }
        }
      }
      return true;
    }

    bool SubmappingInterface::start() {

      LOG(INFO) << "\n\nStarting supereight processing... \n\n";
      finishedIntegratingSensorOnlyCycles_ = 0;
      if (gsMappingEnabled_.load()) {
        size_t free_gpu_bytes = 0;
        size_t total_gpu_bytes = 0;
        const cudaError_t cuda_mem_status = cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes);
        if (cuda_mem_status == cudaSuccess && total_gpu_bytes > 0) {
          gsGpuUsedMemoryStartMb_ =
              static_cast<double>(total_gpu_bytes - free_gpu_bytes) / (1024.0 * 1024.0);
        } else {
          gsGpuUsedMemoryStartMb_ = -1.0;
          LOG(WARNING) << "GS start: cudaMemGetInfo failed: "
                       << cudaGetErrorString(cuda_mem_status);
        }
      }

      {
        std::lock_guard<std::mutex> lock(submapFinalizeMutex_);
        submapFinalizeStop_ = false;
      }
      if (!submapFinalizeWorker_.joinable()) {
        submapFinalizeWorker_ = std::thread(&SubmappingInterface::runSubmapFinalizationWorker, this);
      }

      dataIntegration_ = std::thread(&SubmappingInterface::integrationLoop, this);
      submapIntegration_ = std::thread(&SubmappingInterface::processSupereightFrames, this);

      return true;
    }

    bool SubmappingInterface::stateUpdateCallback(
            const State &latestState, const TrackingState &latestTrackingState,
            std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates) {

      {
        std::lock_guard<std::mutex> lock(mappingFpsMtx_);
        if (!mappingFpsTrackingStartSet_) {
          mappingFpsTrackingStartTime_ = okvis::Time::now();
          mappingFpsTrackingStartSet_ = true;
        }
      }

      {
        std::lock_guard lockCurrentPose(currentPose_mtx_);
        currentPose_ = latestState.T_WS.T().cast<float>();
      }

      if(latestTrackingState.isLidarKeyframe && numIntegratedLidarFrames_ > submapConfig_.submapMinFrames){
        DLOG(INFO) << " LIDAR KF!!! FORCE NEW SUBMAP!" << std::endl;
        lidarKfDetected_ = true;
      }

      std::set<StateId> affectedStateIds;
      auto it = updatedStates->begin();
      while(it != updatedStates->end()){
        affectedStateIds.insert(it->first);
        ++it;
      }

      {
        std::lock_guard _(state_mtx_);
        latestState_ = latestState;
      }

      OkvisUpdate latestStateData(latestState,
                                  latestState.timestamp,
                                  latestTrackingState,
                                  affectedStateIds,
                                  updatedStates);

      if (blocking_) {
        bool success = stateUpdates_.PushBlockingIfFull(latestStateData, 1);
        return success;
      } else {
        if (stateUpdates_.PushNonBlockingDroppingIfFull(latestStateData,
                                                        nonBlockingStateUpdateQueueSize_)) {
          recordLatestWinsDrops("state queue entries",
                                realtimeStateQueueDropCount_, 1);
        }
        return true;
      }
    }


  float SubmappingInterface::evaluateDepthOverlap(const SupereightMapType& map, uint64_t& mapId, SupereightFrames& seFrame) {

    if (seFrame.vecDepthFrames.empty()) {
      LOG(WARNING) << "Depth measurements are empty in supereight frame.";
      return 1.0;
    }

    if (prevKeyframeId_ == UNINITIALIZED_ID) {
      return 1.0;
    }

    Eigen::Isometry3f T_KW = seSubmapLookup_[mapId].T_WK.inverse();
    unsigned int numObserved = 0;
    unsigned int numAll = 0;
    constexpr int skipInterval = 8;
    const float voxResolution = 4.0f * mapConfig_.res;

    for(size_t i = 0; i < seFrame.vecDepthFrames.size(); i++) {
      for(const auto& measurements : seFrame.vecDepthFrames[i]){
        if(viParameters_.nCameraSystem.isDepthCamera(measurements.first)
           || viParameters_.mappingInputEnabledFor(measurements.first)){
          auto& camera = (*cameraSensors_).at(measurements.first).second;
          for(const auto& cam_idx_measurements : measurements.second) {
            if(!cam_idx_measurements.second.measurement.depthImage.empty()) {
              Eigen::Matrix4f T_MD = T_KW * cam_idx_measurements.first.matrix();
              const cv::Mat& depthImage = cam_idx_measurements.second.measurement.depthImage;
              const int depthWidth = depthImage.cols;
              const int depthHeight = depthImage.rows;

              DepthOverlapSamplingCacheKey samplingCacheKey;
              samplingCacheKey.camIdx = static_cast<int>(measurements.first);
              samplingCacheKey.width = depthWidth;
              samplingCacheKey.height = depthHeight;
              samplingCacheKey.skipInterval = skipInterval;

              auto samplingCacheIt = depthOverlapSamplingCache_.find(samplingCacheKey);
              if (samplingCacheIt == depthOverlapSamplingCache_.end()) {
                DepthOverlapSamplingCache samplingCache;
                const int sampleCols = static_cast<int>(
                    std::ceil(static_cast<float>(depthWidth) / skipInterval));
                const int sampleRows = static_cast<int>(
                    std::ceil(static_cast<float>(depthHeight) / skipInterval));
                const int num2D = sampleCols * sampleRows;

                Eigen::Matrix2Xf samplePoints(2, num2D);
                samplingCache.pixelIndices.reserve(num2D);
                int sampleCnt = 0;
                for (int x = 0; x < depthWidth; x += skipInterval) {
                  for (int y = 0; y < depthHeight; y += skipInterval) {
                    samplePoints(0, sampleCnt) = static_cast<float>(x);
                    samplePoints(1, sampleCnt) = static_cast<float>(y);
                    samplingCache.pixelIndices.push_back(y * depthWidth + x);
                    sampleCnt++;
                  }
                }
                samplePoints.conservativeResize(Eigen::NoChange, sampleCnt);

                samplingCache.sampleDirections.resize(3, sampleCnt);
                std::vector<bool> sampleSuccess;
                camera.model.backProjectBatch(samplePoints, &samplingCache.sampleDirections, &sampleSuccess);

                const int dirCols = samplingCache.sampleDirections.cols();
                if (static_cast<int>(samplingCache.pixelIndices.size()) > dirCols) {
                  samplingCache.pixelIndices.resize(dirCols);
                }
                samplingCache.validSampleIndices.reserve(dirCols);

                for (int j = 0; j < dirCols; ++j) {
                  samplingCache.validSampleIndices.push_back(j);
                }

                samplingCacheIt =
                    depthOverlapSamplingCache_.emplace(samplingCacheKey, std::move(samplingCache))
                        .first;
              }

              const auto& samplingCache = samplingCacheIt->second;
              if (samplingCache.validSampleIndices.empty()) {
                continue;
              }

              const float near = camera.near_plane;
              const float far = camera.far_plane;
              const int numSample = static_cast<int>(std::floor((far - near) / voxResolution));
              if (numSample <= 0) {
                continue;
              }
              const int maxSampleIdxGlobal = numSample - 1;

              const Eigen::Matrix3f R_MD = T_MD.topLeftCorner<3,3>();
              const Eigen::Vector3f t_MD = T_MD.topRightCorner<3,1>();
              const Eigen::Matrix3Xf rotatedSampleDirections = R_MD * samplingCache.sampleDirections;
              const float* depthDataPtr = depthImage.ptr<float>(0);

              for (const int sampleIdx : samplingCache.validSampleIndices) {
                const float measDepth_i = depthDataPtr[samplingCache.pixelIndices[sampleIdx]];

                int maxSampleIdx = maxSampleIdxGlobal;
                if (std::isfinite(measDepth_i)) {
                  const float maxSampleIdxFloat =
                      std::floor((measDepth_i - near) / voxResolution);
                  if (maxSampleIdxFloat < 0.0f) {
                    continue;
                  }
                  maxSampleIdx = std::min(
                      maxSampleIdx, static_cast<int>(maxSampleIdxFloat));
                }

                const Eigen::Vector3f rayDir_M = rotatedSampleDirections.col(sampleIdx);
                for (int zi = 0; zi <= maxSampleIdx; ++zi) {
                  const float depth_i = near + voxResolution * static_cast<float>(zi);
                  const Eigen::Vector3f p_M = rayDir_M * depth_i + t_MD;

                  std::optional<float> occ = map.interpField<se::Safe::On>(p_M);
                  numAll++;
                  if (occ){
                    numObserved ++;
                  }
                }
              }
            }
          }
        }
      }
    }

    if(numAll == 0) {
      return 1.0;
    }

    return float(numObserved) / numAll;
  }

  std::vector<uint64_t> SubmappingInterface::collectCovisibleOlderSubmaps(
      const uint64_t activeId,
      const uint64_t previousId,
      const int recursionDepth)
  {
    std::vector<uint64_t> candidateIds;
    if (activeId == UNINITIALIZED_ID || previousId == UNINITIALIZED_ID || activeId == previousId) {
      return candidateIds;
    }

    std::vector<uint64_t> covisibleIds;
    obtainCovisibles(covisibleIds, StateId(activeId), recursionDepth);
    candidateIds.reserve(covisibleIds.size());
    for (const uint64_t covisibleId : covisibleIds) {
      if (covisibleId >= previousId) {
        continue;
      }
      const auto it = seSubmapLookup_.find(covisibleId);
      if (it != seSubmapLookup_.end() && it->second.map) {
        candidateIds.push_back(covisibleId);
      }
    }
    std::sort(candidateIds.begin(), candidateIds.end());
    candidateIds.erase(std::unique(candidateIds.begin(), candidateIds.end()), candidateIds.end());
    return candidateIds;
  }

  std::vector<std::pair<float, uint64_t>> SubmappingInterface::rankSubmapCandidatesByIou(
      const uint64_t activeId,
      const std::vector<uint64_t>& candidateIds,
      const size_t topK,
      const float minIou) const
  {
    std::vector<std::pair<float, uint64_t>> rankedCandidates;
    const auto activeIt = seSubmapLookup_.find(activeId);
    if (activeIt == seSubmapLookup_.end() || !activeIt->second.map) {
      return rankedCandidates;
    }

    const Eigen::Vector3f bbox_active_min = activeIt->second.map->aabb().min();
    const Eigen::Vector3f bbox_active_max = activeIt->second.map->aabb().max();
    rankedCandidates.reserve(candidateIds.size());

    for (const uint64_t candidateId : candidateIds) {
      const auto candidateIt = seSubmapLookup_.find(candidateId);
      if (candidateIt == seSubmapLookup_.end() || !candidateIt->second.map) {
        continue;
      }

      Eigen::Vector3f bbox_test_max;
      Eigen::Vector3f bbox_test_min;
      get_bbox(candidateIt->second,
               activeIt->second.T_WK.inverse().matrix() * candidateIt->second.T_WK.matrix(),
               bbox_test_max,
               bbox_test_min);
      const float iou =
          calculateIoU3D(bbox_active_min, bbox_active_max, bbox_test_min, bbox_test_max);
      if (iou > minIou) {
        rankedCandidates.emplace_back(iou, candidateId);
      }
    }

    std::stable_sort(rankedCandidates.begin(),
                     rankedCandidates.end(),
                     [](const auto& lhs, const auto& rhs) {
                       if (lhs.first != rhs.first) {
                         return lhs.first > rhs.first;
                       }
                       return lhs.second < rhs.second;
                     });
    if (rankedCandidates.size() > topK) {
      rankedCandidates.resize(topK);
    }
    return rankedCandidates;
  }

  std::vector<uint64_t> SubmappingInterface::verifySubmapCandidatesByObservedPoints(
      const uint64_t activeId,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& queryPointsB,
      const std::vector<std::pair<float, uint64_t>>& rankedCandidates,
      const size_t minObservedPoints) const
  {
    std::vector<uint64_t> verifiedIds;
    if (activeId == UNINITIALIZED_ID || queryPointsB.empty()) {
      return verifiedIds;
    }

    verifiedIds.reserve(rankedCandidates.size());
    for (const auto& candidate : rankedCandidates) {
      uint64_t testId = candidate.second;
      uint64_t activeIdMutable = activeId;
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> tmp;
      const size_t observed_points =
          determineObservedPoints(testId, activeIdMutable, queryPointsB, tmp);
      if (observed_points > minObservedPoints) {
        verifiedIds.push_back(testId);
      }
    }
    return verifiedIds;
  }

  SubmappingInterface::SubmapOverlapCandidates SubmappingInterface::collectOverlappingSubmaps(
      const uint64_t activeId,
      const uint64_t previousId,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& queryPointsB)
  {
    SubmapOverlapCandidates result;
    result.covisible_older_candidates =
        collectCovisibleOlderSubmaps(activeId, previousId, 15);
    result.iou_ranked_candidates = rankSubmapCandidatesByIou(
        activeId, result.covisible_older_candidates, 5, 0.2f);
    const size_t minObservedPoints = static_cast<size_t>(0.5f * submapConfig_.numSubmapFactors);
    result.verified_ranked_ids = verifySubmapCandidatesByObservedPoints(
        activeId, queryPointsB, result.iou_ranked_candidates, minObservedPoints);
    result.verified_oldest_first_ids = result.verified_ranked_ids;
    std::sort(result.verified_oldest_first_ids.begin(), result.verified_oldest_first_ids.end());
    result.verified_oldest_first_ids.erase(
        std::unique(result.verified_oldest_first_ids.begin(),
                    result.verified_oldest_first_ids.end()),
        result.verified_oldest_first_ids.end());
    return result;
  }

  uint64_t SubmappingInterface::findOldestOverlappingSubmap(
          std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &downsampledPointCloud,
          uint64_t activeId, uint64_t previousId) {
    const SubmapOverlapCandidates result =
        collectOverlappingSubmaps(activeId, previousId, downsampledPointCloud);
    if (!result.verified_oldest_first_ids.empty()) {
      return result.verified_oldest_first_ids.front();
    }
    return std::numeric_limits<uint64_t>::max();
  }

  size_t SubmappingInterface::determineObservedPoints(uint64_t& mapIdA, uint64_t& mapIdB,
                                                      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints) const {
    size_t observed_counter = 0;
    size_t number_of_measurements = points.size();
    const auto& submapA = seSubmapLookup_.at(mapIdA);
    const auto& submapB = seSubmapLookup_.at(mapIdB);
    okvis::kinematics::Transformation T_WM_A(submapA.T_WK.matrix().cast<double>());
    okvis::kinematics::Transformation T_WM_B(submapB.T_WK.matrix().cast<double>());

    for(size_t i = 0; i < number_of_measurements; i++) {

      Eigen::Vector3f pt_A = T_WM_A.inverse().T3x4().cast<float>() * T_WM_B.T().cast<float>() * points.at(i).homogeneous();
      std::optional<float> occ = submapA.map->interpField<se::Safe::On>(pt_A);
      if (occ){
        observedPoints.push_back(points.at(i));
        observed_counter++;
      }
    }
    return observed_counter;
  }

  size_t SubmappingInterface::determineObservedPointsAndSigma(uint64_t& mapIdA, uint64_t& mapIdB,
                                                              const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                                              std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints,
                                                              const std::vector<float>& sigma, std::vector<float>& observedSigma) const {
    size_t observed_counter = 0;
    size_t number_of_measurements = points.size();
    const float occ_margin = 0.8*dataConfig_.field.log_odd_max;
    const auto& submapA = seSubmapLookup_.at(mapIdA);
    const auto& submapB = seSubmapLookup_.at(mapIdB);
    okvis::kinematics::Transformation T_WM_A(submapA.T_WK.matrix().cast<double>());
    okvis::kinematics::Transformation T_WM_B(submapB.T_WK.matrix().cast<double>());

    for(size_t i = 0; i < number_of_measurements; i++) {

      Eigen::Vector3f pt_A = T_WM_A.inverse().T3x4().cast<float>() * T_WM_B.T().cast<float>() * points.at(i).homogeneous();
      std::optional<float> occ =
          okvis::interpFieldMeanOccup<se::Safe::On>(*(submapA.map), pt_A);
      std::optional<Eigen::Vector3f> gradf =
          okvis::gradFieldMeanOccup<se::Safe::On>(*(submapA.map), pt_A);
      if (occ && gradf){
        if (occ.value() > -occ_margin && occ.value() < occ_margin && gradf.value().norm() >= 1e-03) {
          observedPoints.push_back(points.at(i));
          observedSigma.push_back(sigma.at(i));
          observed_counter++;
        }
      }
    }
    return observed_counter;
  }

  bool SubmappingInterface::decideNewSubmap(SupereightFrames& supereightFrame) {
    bool create_new_submap = false;
    if(cameraSensors_){
      if(prevKeyframeId_ == UNINITIALIZED_ID){
        create_new_submap = true;
      }
      else if(supereightFrame.keyFrameId != prevKeyframeId_){
        if (supereightFrame.keyFrameId != curKeyframeId_) {
          keyframeCounter_ ++; 
          curKeyframeId_ = supereightFrame.keyFrameId;
        }
        float overlap = 1.0;
        if(submapCounter_ > 1){
          overlap = evaluateDepthOverlap(*(seSubmapLookup_[previousSubmapId_].map), previousSubmapId_, supereightFrame);
        }
        DLOG(INFO) << "overlap = " << overlap << ", " << submapConfig_.submapOverlapRatio
          << ", keyframeCounter_ = " << keyframeCounter_<< ", " << submapConfig_.submapKfThreshold
          << ", numIntegratedDepthFrames_ = " << numIntegratedDepthFrames_ << ", " << submapConfig_.submapMinFrames;
        if ((overlap < submapConfig_.submapOverlapRatio || keyframeCounter_ > submapConfig_.submapKfThreshold)
            && numIntegratedDepthFrames_ > submapConfig_.submapMinFrames) {
          keyframeCounter_ = 0;
          create_new_submap = true;
        }
        else {
          supereightFrame.keyFrameId = prevKeyframeId_;
        }
      }
    }
    else 

    DLOG(INFO) << "curKeyframeId_ = " << curKeyframeId_
      << ", supereightFrame.keyFrameId = " << supereightFrame.keyFrameId
      << ", prevKeyframeId_ = " << prevKeyframeId_
      << ", submapCounter_ = " << submapCounter_ 
      << ", previousSubmapId_ = " << previousSubmapId_;

    return create_new_submap;
  }

  void SubmappingInterface::addSubmapAlignmentFactors(const SupereightFrames& supereightFrame) {
    if (submapConfig_.useMap2MapFactors) {
      if (submapCounter_ > 2) {
        if (cameraSensors_) {
          depthMap2MapFactors();
        }
      }
      else{
        if(alignCallback_ && previousSubmap_) {
          DLOG(INFO) << "Calling callback with previousSubmapId_ = " << previousSubmapId_
                    << "and " << submapAlignBlock_.pointCloud_B.size() << " points";
          std::vector<float> zeros(submapAlignBlock_.pointCloud_B.size());
          std::fill(zeros.begin(), zeros.end(), 0.0f);
          alignCallback_(nullptr, seSubmapLookup_[previousSubmapId_].map.get(), 0, previousSubmapId_,
                         submapAlignBlock_.pointCloud_B, zeros);
        }
      }

      submapAlignBlock_.frame_A_id = prevKeyframeId_;
      submapAlignBlock_.frame_B_id = supereightFrame.keyFrameId;
      submapAlignBlock_.pointCloud_B.clear();
      submapAlignBlock_.sigma_B.clear();
      submapAlignBlock_.voxelHashMap.Clear();
      DLOG(INFO) << "Created new SubmapAlignmentTerm with frame A id: " << prevKeyframeId_ 
                << " and frame B id: " << supereightFrame.keyFrameId;
    }
    else if(submapConfig_.useMap2LiveFactors){
        if(previousSubmap_){
            std::vector<float> zeros;
            alignCallback_(nullptr, seSubmapLookup_[previousSubmapId_].map.get(), 0, previousSubmapId_,
                           submapAlignBlock_.pointCloud_B, zeros);
        }
    }
  }

  void SubmappingInterface::depthMap2MapFactors() {
    if(submapAlignBlock_.pointCloud_B.size() > 0){
      uint64_t selected_submap_id = findOldestOverlappingSubmap(submapAlignBlock_.pointCloud_B, submapAlignBlock_.frame_B_id, submapAlignBlock_.frame_A_id);
      if(selected_submap_id == std::numeric_limits<uint64_t>::max()) {
        selected_submap_id = submapAlignBlock_.frame_A_id;
      }

      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPoints;
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> downsampledPointCloud;
      std::vector<float> observedSigmas, downsampledSigmas;

      determineObservedPointsAndSigma(selected_submap_id, submapAlignBlock_.frame_B_id,
                  submapAlignBlock_.pointCloud_B, observedPoints,
                  submapAlignBlock_.sigma_B, observedSigmas);
      if (observedPoints.size() > 5*submapConfig_.numSubmapFactors) {
        okvis::downsamplePointsUncertainty(observedPoints, observedSigmas, downsampledPointCloud, downsampledSigmas, 5*submapConfig_.numSubmapFactors);
      }
      else if (observedPoints.size() > 0) {
        downsampledPointCloud = observedPoints;
        downsampledSigmas = observedSigmas;
      }
      if (downsampledPointCloud.size() > 0) {
        alignCallback_(seSubmapLookup_[selected_submap_id].map.get(), seSubmapLookup_[submapAlignBlock_.frame_B_id].map.get(),
                        selected_submap_id,
                        submapAlignBlock_.frame_B_id,
                        downsampledPointCloud, downsampledSigmas);


        float sigmaMean = 0;
        for (size_t ii = 0; ii < downsampledSigmas.size(); ii++) {
          sigmaMean += downsampledSigmas[ii];
        }

        DLOG(INFO) << "[Depth: Map-to-map] Overlapping submap ids: "
          << selected_submap_id << "-" << submapAlignBlock_.frame_B_id
          << " with " << observedPoints.size()
          << " overlap points --> " << downsampledPointCloud.size()
          << " with uncertainty " << sigmaMean / downsampledSigmas.size();
      }
    }
  }

  void SubmappingInterface::updateDepthAlignBlock(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
    const kinematics::Transformation& T_WK, const int& depthImage_idx) {

    int depthWidth = depthData.second.measurement.depthImage.cols;
    int depthHeight = depthData.second.measurement.depthImage.rows;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> depthRays(depthWidth * depthHeight);
    std::vector<float> depthSigmas(depthWidth * depthHeight);
    Eigen::Matrix4f T_MD = T_WK.inverse().T().cast<float>() * depthData.first.matrix();

    size_t counter_xy = 0;
    for (int x = 0; x < depthWidth; x+=20) {
      for (int y = 0; y < depthHeight; y+=20) {

        float pointZ = depthData.second.measurement.depthImage.at<float>(y,x);
        float sigmaZ = submapConfig_.sensorError;

        Eigen::Vector2f imagePoint_xy(static_cast<float>(x), static_cast<float>(y));
        Eigen::Vector3f cameraPoint;
        (*cameraSensors_).at(depthImage_idx).second.model.backProject(imagePoint_xy, &cameraPoint);
        cameraPoint = cameraPoint * pointZ;
        depthRays[counter_xy] = T_MD.block<3,3>(0,0) * cameraPoint + T_MD.block<3,1>(0,3);
        depthSigmas[counter_xy] = sigmaZ;
        counter_xy++;
      }
    }
    depthRays.resize(counter_xy);
    depthSigmas.resize(counter_xy);
    submapAlignBlock_.pointCloud_B.insert(submapAlignBlock_.pointCloud_B.end(), depthRays.begin(), depthRays.end());
    submapAlignBlock_.sigma_B.insert(submapAlignBlock_.sigma_B.end(), depthSigmas.begin(), depthSigmas.end());
  }

}

namespace okvis {
void SubmappingInterface::saveAllSubmapMeshes() {
  waitForSubmapFinalization();
  for (const auto& submap : seSubmapLookup_) saveSubmap(submap.first);
}
}
