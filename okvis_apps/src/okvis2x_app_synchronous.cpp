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

/**
 * @file okvis2x_app_synchronous.cpp
 * @brief Dataset SLAM and RGB-D mapping application
 * @author Simon Boche
 * @author Jaehyung Jung
 * @author Sebastian Barbas Laina
 */

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <array>
#include <fstream>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <opencv2/highgui/highgui.hpp>

#include <okvis/XDatasetReader.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <okvis/SubmappingInterface.hpp>
#include "OkvisOccupancyViewer3D.hpp"

namespace {

;

;

std::string trimCsvToken(const std::string& input) {
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
  while (end > begin && input[end - 1] == ',') {
    --end;
  }
  return input.substr(begin, end - begin);
}

okvis::Time timeFromNs(const uint64_t timestamp_ns) {
  const uint32_t sec = static_cast<uint32_t>(timestamp_ns / 1000000000ULL);
  const uint32_t nsec = static_cast<uint32_t>(timestamp_ns % 1000000000ULL);
  return okvis::Time(sec, nsec);
}

bool loadDatasetTimestampIndex(const std::string& csv_path,
                               std::vector<uint64_t>& timestamps_out) {
  timestamps_out.clear();
  std::ifstream in(csv_path);
  if (!in.good()) {
    return false;
  }

  std::string line;
  std::getline(in, line);  
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string ts_str;
    if (!std::getline(ss, ts_str, ',')) {
      continue;
    }
    ts_str = trimCsvToken(ts_str);
    if (ts_str.empty()) {
      continue;
    }
    try {
      timestamps_out.push_back(static_cast<uint64_t>(std::stoull(ts_str)));
    } catch (...) {
      continue;
    }
  }
  return !timestamps_out.empty();
}

bool timestampsIdentical(const std::vector<uint64_t>& lhs,
                         const std::vector<uint64_t>& rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool writeTrajectoryCsv(
    const std::string& csv_path,
    const std::vector<okvis::ViSlamBackend::FinalTrajectoryPosition>& poses) {
  const boost::filesystem::path output_path(csv_path);
  if (!output_path.parent_path().empty()) {
    boost::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(csv_path);
  if (!out.is_open()) {
    return false;
  }

  out << "timestamp, p_WS_W_x, p_WS_W_y, p_WS_W_z, q_WS_x, q_WS_y, q_WS_z, q_WS_w\n";
  out << std::scientific << std::setprecision(18);
  for (const auto& pose : poses) {
    out << pose.timestamp_ns << ", "
        << pose.px << ", "
        << pose.py << ", "
        << pose.pz << ", "
        << pose.qx << ", "
        << pose.qy << ", "
        << pose.qz << ", "
        << pose.qw << "\n";
  }
  return out.good();
}

using GtPoses = std::map<uint64_t, okvis::kinematics::Transformation>;

bool loadGtPoses(
    const std::string& filename, GtPoses& poses,
    std::map<uint64_t, okvis::ViSlamBackend::FinalTrajectoryPosition>& trajectory) {
  std::ifstream input(filename);
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    okvis::ViSlamBackend::FinalTrajectoryPosition pose{};
    if (!(row >> pose.timestamp_ns >> pose.px >> pose.py >> pose.pz
              >> pose.qx >> pose.qy >> pose.qz >> pose.qw)) {
      return false;
    }
    const Eigen::Vector3d position(pose.px, pose.py, pose.pz);
    Eigen::Quaterniond rotation(pose.qw, pose.qx, pose.qy, pose.qz);
    if (!position.allFinite() || !rotation.coeffs().allFinite()
        || rotation.norm() < 1e-12) {
      return false;
    }
    rotation.normalize();
    poses.emplace(pose.timestamp_ns,
                  okvis::kinematics::Transformation(position, rotation));
    trajectory.emplace(pose.timestamp_ns, pose);
  }
  return !poses.empty();
}

bool parseBoolNode(const cv::FileNode& node, const bool default_value) {
  if (node.empty()) {
    return default_value;
  }
  if (node.isInt() || node.isReal()) {
    return static_cast<int>(node) != 0;
  }
  if (node.isString()) {
    std::string value;
    node >> value;
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "true" || value == "yes" || value == "on" || value == "1") {
      return true;
    }
    if (value == "false" || value == "no" || value == "off" || value == "0") {
      return false;
    }
    return default_value;
  }
  return default_value;
}

std::vector<float> parseFloatListNode(const cv::FileNode& node)
{
  std::vector<float> values;
  if (node.type() != cv::FileNode::SEQ) {
    return values;
  }
  for (const auto& entry : node) {
    if (entry.isInt() || entry.isReal()) {
      values.push_back(static_cast<float>(entry.real()));
    }
  }
  return values;
}

std::vector<std::string> parseStringListNode(const cv::FileNode& node)
{
  std::vector<std::string> values;
  if (node.type() != cv::FileNode::SEQ) {
    return values;
  }
  for (const auto& entry : node) {
    if (entry.isString()) {
      std::string value;
      entry >> value;
      if (!value.empty()) {
        values.push_back(value);
      }
    }
  }
  return values;
}

bool appendGsRunSummaryCsv(const std::string& save_path,
                           const std::string& mode,
                           const double total_processing_sec,
                           const okvis::GsFinalizationSummary& gs_summary) {
  const boost::filesystem::path summary_path =
      boost::filesystem::path(save_path) / ("okvis2-" + mode + "-run_summary.csv");
  if (!summary_path.parent_path().empty()) {
    boost::filesystem::create_directories(summary_path.parent_path());
  }

  const bool need_header =
      !boost::filesystem::exists(summary_path) || boost::filesystem::file_size(summary_path) == 0;
  std::ofstream out(summary_path.string(), std::ios::out | std::ios::app);
  if (!out.is_open()) {
    return false;
  }

  if (need_header) {
    out << "total_processing_time_sec,"
        << "avg_fps,"
        << "global_opt_time_sec,"
        << "slam_time_wo_global_opt_sec,"
        << "gpu_memory_usage_mb,"
        << "keyframes,"
        << "save_iteration_frames,"
        << "final_global_gaussian_count\n";
  }

  const double slam_without_global_opt_sec =
      std::max(0.0, total_processing_sec - gs_summary.global_optimization_time_sec);
  out << std::fixed << std::setprecision(6)
      << total_processing_sec << ","
      << gs_summary.pre_global_opt_system_fps << ","
      << gs_summary.global_optimization_time_sec << ","
      << slam_without_global_opt_sec << ","
      << gs_summary.gpu_memory_usage_mb << ","
      << gs_summary.global_opt_kf_count << ","
      << gs_summary.save_iteration_frames << ","
      << gs_summary.final_global_gaussian_count << "\n";
  return out.good();
}

size_t findMappingCameraId(const okvis::ViParameters& parameters) {
  if (parameters.mappingInputPreprocessor
      && parameters.mappingInputPreprocessor->enabled()) {
    return parameters.mappingInputPreprocessor->targetCameraIndex();
  }
  for (size_t cam_idx = 0; cam_idx < parameters.nCameraSystem.numCameras(); ++cam_idx) {
    if (parameters.nCameraSystem.cameraType(cam_idx).isUsedMapping) {
      return cam_idx;
    }
  }
  return std::numeric_limits<size_t>::max();
}

float sampledMedianDepth(const cv::Mat& depth_image) {
  if (depth_image.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  cv::Mat depth_float = depth_image;
  if (depth_float.type() != CV_32FC1) {
    depth_image.convertTo(depth_float, CV_32FC1);
  }
  if (!depth_float.isContinuous()) {
    depth_float = depth_float.clone();
  }

  const int row_step = std::max(1, depth_float.rows / 64);
  const int col_step = std::max(1, depth_float.cols / 64);
  std::vector<float> samples;
  samples.reserve(((depth_float.rows + row_step - 1) / row_step)
                  * ((depth_float.cols + col_step - 1) / col_step));

  for (int y = 0; y < depth_float.rows; y += row_step) {
    const float* row_ptr = depth_float.ptr<float>(y);
    for (int x = 0; x < depth_float.cols; x += col_step) {
      const float depth = row_ptr[x];
      if (std::isfinite(depth) && depth > 0.0f) {
        samples.push_back(depth);
      }
    }
  }

  if (samples.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  auto middle = samples.begin() + samples.size() / 2;
  std::nth_element(samples.begin(), middle, samples.end());
  return *middle;
}

void renderPreviewTile(cv::Mat tile, const cv::Mat& image, const std::string& label) {
  tile.setTo(cv::Scalar(24, 24, 24));
  if (!image.empty() && image.cols > 0 && image.rows > 0) {
    const double scale = std::min(static_cast<double>(tile.cols) / static_cast<double>(image.cols),
                                  static_cast<double>(tile.rows) / static_cast<double>(image.rows));
    const int resized_w = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat display;
    if (resized.channels() == 3) {
      cv::cvtColor(resized, display, cv::COLOR_RGB2BGR);
    } else if (resized.channels() == 1) {
      cv::cvtColor(resized, display, cv::COLOR_GRAY2BGR);
    } else {
      display = resized;
    }
    const int x = (tile.cols - resized_w) / 2;
    const int y = (tile.rows - resized_h) / 2;
    display.copyTo(tile(cv::Rect(x, y, resized_w, resized_h)));
  }
  cv::putText(tile,
              label,
              cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX,
              0.75,
              cv::Scalar(235, 235, 235),
              2,
              cv::LINE_AA);
}

void renderInfoTile(cv::Mat info,
                    const okvis::GsPreviewFrame& gs_preview,
                    const uint64_t mapped_frame_count,
                    const uint64_t input_frame_count) {
  info.setTo(cv::Scalar(18, 18, 18));
  const auto fmtFps = [](const float value) {
    return std::string(cv::format("%.2f", static_cast<double>(value)));
  };
  constexpr int kLabelX = 12;
  constexpr int kValueX = 220;
  constexpr int kFirstRowY = 76;
  constexpr int kRowStep = 32;
  const auto drawStatRow = [&](const int row,
                               const std::string& label,
                               const std::string& value,
                               const cv::Scalar& colour) {
    const int y = kFirstRowY + row * kRowStep;
    cv::putText(info, label, cv::Point(kLabelX, y), cv::FONT_HERSHEY_SIMPLEX,
                0.75, colour, 2, cv::LINE_AA);
    cv::putText(info, value, cv::Point(kValueX, y), cv::FONT_HERSHEY_SIMPLEX,
                0.75, colour, 2, cv::LINE_AA);
  };
  cv::putText(info, "GS Stats", cv::Point(12, 34), cv::FONT_HERSHEY_SIMPLEX, 0.9,
              cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
  const cv::Scalar primary_colour(200, 200, 200);
  drawStatRow(0, "Frames:",
              std::to_string(mapped_frame_count) + "/"
                  + std::to_string(input_frame_count),
              primary_colour);
  drawStatRow(1, "Splats:", std::to_string(gs_preview.num_splats), primary_colour);
  drawStatRow(2, "Keyframes:", std::to_string(gs_preview.num_kf), primary_colour);
  drawStatRow(3, "Submaps:", std::to_string(gs_preview.num_submaps), primary_colour);
  if (gs_preview.global_iter > 0) {
    drawStatRow(4, "Global Iter:", std::to_string(gs_preview.global_iter),
                cv::Scalar(210, 230, 180));
  }
  drawStatRow(5, "FPS(Input):", fmtFps(gs_preview.input_fps), primary_colour);
  drawStatRow(6, "FPS(Tracking):", fmtFps(gs_preview.tracking_fps), primary_colour);
  drawStatRow(7, "FPS(System):", fmtFps(gs_preview.fps), primary_colour);
  drawStatRow(8, "FPS(Backend):", fmtFps(gs_preview.backend_fps),
              cv::Scalar(180, 180, 180));
}

cv::Mat composeGoMappingWindow(const okvis::GsPreviewFrame& gs_preview,
                               const uint64_t mapped_frame_count,
                               const uint64_t input_frame_count) {
  static thread_local std::array<cv::Mat, 2> composed_buffers = {
      cv::Mat(720, 1280, CV_8UC3),
      cv::Mat(720, 1280, CV_8UC3)};
  static thread_local size_t write_index = 0;

  cv::Mat& composed = composed_buffers[write_index];
  if (composed.rows != 720 || composed.cols != 1280 || composed.type() != CV_8UC3) {
    composed = cv::Mat(720, 1280, CV_8UC3);
  }

  renderPreviewTile(composed(cv::Rect(0, 0, 640, 360)), gs_preview.rgb, "Input RGB");
  renderPreviewTile(
      composed(cv::Rect(640, 0, 640, 360)), gs_preview.rendered_rgb, "Rendered RGB");
  renderPreviewTile(composed(cv::Rect(0, 360, 640, 360)), gs_preview.depth, "Input Depth");
  {
    renderInfoTile(
        composed(cv::Rect(640, 360, 640, 360)),
        gs_preview,
        mapped_frame_count,
        input_frame_count);
  }
  write_index = (write_index + 1) % composed_buffers.size();
  return composed;
}

bool windowVisible(const std::string& window_name) {
  const double visible = cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE);
  return visible >= 1.0;
}

int64_t steadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct FpsSampleState {
  uint64_t last_count = 0;
  int64_t last_sample_ns = 0;
  float filtered_hz = 0.0f;
};

float sampleRateHz(const uint64_t count, FpsSampleState& state) {
  const int64_t now_ns = steadyNowNs();
  if (state.last_sample_ns <= 0) {
    state.last_sample_ns = now_ns;
    state.last_count = count;
    return state.filtered_hz;
  }

  const int64_t dt_ns = now_ns - state.last_sample_ns;
  if (dt_ns <= 0) {
    return state.filtered_hz;
  }
  const double dt_sec = static_cast<double>(dt_ns) * 1.0e-9;
  if (dt_sec <= 1.0e-6) {
    return state.filtered_hz;
  }

  const uint64_t delta_count = count - state.last_count;
  const float inst_hz = static_cast<float>(static_cast<double>(delta_count) / dt_sec);
  constexpr float kAlpha = 0.35f;
  state.filtered_hz =
      (state.filtered_hz > 0.0f)
          ? ((1.0f - kAlpha) * state.filtered_hz + kAlpha * inst_hz)
          : inst_hz;
  state.last_sample_ns = now_ns;
  state.last_count = count;
  return state.filtered_hz;
}
}  

int main(int argc, char **argv)
{

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;
  FLAGS_log_prefix = 0;

  bool headless = false;
  bool useGtPoses = false;
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--headless") {
      headless = true;
    } else if (std::string(argv[i]) == "--gt-poses") {
      useGtPoses = true;
    } else {
      args.emplace_back(argv[i]);
    }
  }
  if (args.size() != 3 && args.size() != 4) {
    LOG(ERROR) << "Usage: " << argv[0]
               << " [--headless] [--gt-poses] [config-okvis2.yaml] [config-se2.yaml] [input-directory] "
                  "(optional)[output-directory]";
    return EXIT_FAILURE;
  }

  std::string configFilename(args[0]);
  std::string seConfigFilename(args[1]);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  se::SubMapConfig submapConfig(seConfigFilename);
  parameters.mappingInputDownsampling =
      static_cast<size_t>(submapConfig.depthImageResDownsampling);

  okvis::GsDensifyConfig gsDensifyConfig;

  okvis::OccupancyViewerConfig occupancyViewerConfig;
  {
    cv::FileStorage okvisFs;
    if (okvisFs.open(configFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML)) {
      const cv::FileNode gsNode = okvisFs["gs_mapping_parameters"];
      if (gsNode.type() == cv::FileNode::MAP) {
        const cv::FileNode saveNode = gsNode["save_rendered_rgb"];
        if (!saveNode.empty()) {
          submapConfig.gsSaveRenderedRgb =
              parseBoolNode(saveNode, submapConfig.gsSaveRenderedRgb);
        }
        const cv::FileNode renderDirNode = gsNode["rendered_rgb_directory"];
        if (!renderDirNode.empty()) {
          std::string renderDir;
          renderDirNode >> renderDir;
          if (!renderDir.empty()) {
            submapConfig.gsRenderedRgbDirectory = renderDir;
          }
        }

        const cv::FileNode densifyNode = gsNode["densify"];
        if (densifyNode.type() == cv::FileNode::MAP) {
          if (!densifyNode["enabled"].empty())
            gsDensifyConfig.enabled = parseBoolNode(densifyNode["enabled"], gsDensifyConfig.enabled);
          if (!densifyNode["interval"].empty())
            gsDensifyConfig.interval = static_cast<int>(densifyNode["interval"]);
          if (!densifyNode["tile_size"].empty())
            gsDensifyConfig.tile_size = static_cast<int>(densifyNode["tile_size"]);
          if (!densifyNode["top_tiles_ratio"].empty())
            gsDensifyConfig.top_tiles_ratio = static_cast<float>(densifyNode["top_tiles_ratio"].real());
          if (!densifyNode["top_peaks_ratio"].empty())
            gsDensifyConfig.top_peaks_ratio = static_cast<float>(densifyNode["top_peaks_ratio"].real());
          if (!densifyNode["max_new_gaussians"].empty())
            gsDensifyConfig.max_new_gaussians = static_cast<int>(densifyNode["max_new_gaussians"]);
          if (!densifyNode["init_scale_mul"].empty())
            gsDensifyConfig.init_scale_mul = static_cast<float>(densifyNode["init_scale_mul"].real());
          if (!densifyNode["init_z_scale"].empty())
            gsDensifyConfig.init_z_scale = static_cast<float>(densifyNode["init_z_scale"].real());
          if (!densifyNode["split_size_ratio"].empty())
            gsDensifyConfig.split_size_ratio = static_cast<float>(densifyNode["split_size_ratio"].real());
          if (!densifyNode["split_factor"].empty())
            gsDensifyConfig.split_factor = static_cast<int>(densifyNode["split_factor"]);
          if (!densifyNode["split_scale_shrink"].empty())
            gsDensifyConfig.split_scale_shrink = static_cast<float>(densifyNode["split_scale_shrink"].real());
        }

      }
      const cv::FileNode visNode = okvisFs["visualization_parameters"];
      if (visNode.type() == cv::FileNode::MAP) {
        const cv::FileNode enableOccViewerNode = visNode["enable_occupancy_viewer"];
        if (!enableOccViewerNode.empty()) {
          occupancyViewerConfig.enable =
              parseBoolNode(enableOccViewerNode, occupancyViewerConfig.enable);
        }
        const cv::FileNode occUpdateIntervalNode = visNode["occupancy_viewer_update_interval"];
        if (!occUpdateIntervalNode.empty()) {
          occupancyViewerConfig.update_interval = std::max(1, static_cast<int>(occUpdateIntervalNode));
        }
        const cv::FileNode occMaxFpsNode = visNode["occupancy_viewer_max_fps"];
        if (!occMaxFpsNode.empty()) {
          occupancyViewerConfig.max_fps = std::max(1, static_cast<int>(occMaxFpsNode));
        }
        const cv::FileNode occVoxelStrideNode = visNode["occupancy_viewer_voxel_stride"];
        if (!occVoxelStrideNode.empty()) {
          occupancyViewerConfig.voxel_stride = std::max(1, static_cast<int>(occVoxelStrideNode));
        }
        const cv::FileNode occMaxPointsNode = visNode["occupancy_viewer_max_points"];
        if (!occMaxPointsNode.empty()) {
          occupancyViewerConfig.max_points = std::max(1000, static_cast<int>(occMaxPointsNode));
        }

        const cv::FileNode occOccupiedPointSizeNode = visNode["occupancy_viewer_occupied_point_size"];
        if (!occOccupiedPointSizeNode.empty()) {
          occupancyViewerConfig.occupied_point_size =
              std::max(1, static_cast<int>(occOccupiedPointSizeNode));
        }

        const cv::FileNode occShowPoseNode = visNode["occupancy_viewer_show_pose"];
        if (!occShowPoseNode.empty()) {
          occupancyViewerConfig.show_pose =
              parseBoolNode(occShowPoseNode, occupancyViewerConfig.show_pose);
        }
        const cv::FileNode occWindowNameNode = visNode["occupancy_viewer_window_name"];
        if (!occWindowNameNode.empty()) {
          std::string windowName;
          occWindowNameNode >> windowName;
          if (!windowName.empty()) {
            occupancyViewerConfig.window_name = windowName;
          }
        }
        const cv::FileNode occWindowWidthNode = visNode["occupancy_viewer_window_width"];
        if (!occWindowWidthNode.empty()) {
          occupancyViewerConfig.window_width =
              std::max(1, static_cast<int>(occWindowWidthNode));
        }
        const cv::FileNode occWindowHeightNode = visNode["occupancy_viewer_window_height"];
        if (!occWindowHeightNode.empty()) {
          occupancyViewerConfig.window_height =
              std::max(1, static_cast<int>(occWindowHeightNode));
        }

        const cv::FileNode fisherUpdateIntervalNode =
            visNode["fisher_visualization_update_interval"];
        if (!fisherUpdateIntervalNode.empty()) {
          occupancyViewerConfig.fisher_update_interval =
              std::max(1, static_cast<int>(fisherUpdateIntervalNode));
        }
        const cv::FileNode fisherDisplayRegularizerNode =
            visNode["fisher_display_regularizer"];
        if (!fisherDisplayRegularizerNode.empty()) {
          occupancyViewerConfig.fisher_display_regularizer =
              std::max(1.0e-6f, static_cast<float>(fisherDisplayRegularizerNode.real()));
        }
        const cv::FileNode fisherEigRegularizerNode =
            visNode["fisher_eig_regularizer"];
        if (!fisherEigRegularizerNode.empty()) {
          occupancyViewerConfig.fisher_eig_regularizer =
              std::max(1.0e-12f, static_cast<float>(fisherEigRegularizerNode.real()));
        }
      }
      okvisFs.release();
    }
  }
  if (headless) {
    occupancyViewerConfig.enable = false;
    submapConfig.gsShowPreview = false;
    parameters.output.display_topview = false;
    parameters.output.display_matches = false;
    parameters.output.display_overhead = false;
    LOG(INFO) << "Headless mode: all GUI output disabled.";
  }

  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataType::Config dataConfig;
  dataConfig.readYaml(seConfigFilename);
  const bool isDepth = true;
  const bool isLidar = !isDepth;
  const bool isSubmapping = parameters.output.enable_submapping;

  const bool realtimePipelineEnabled = submapConfig.realtimePipelineEnabled;
  const double inputTargetFps =
      (submapConfig.inputTargetFps > 0.0) ? submapConfig.inputTargetFps : 60.0;

  if (realtimePipelineEnabled) {

    submapConfig.depthQueueDropIfFull = true;
  }

  std::string path(args[2]);

  std::shared_ptr<const GtPoses> gtPoses;
  std::map<uint64_t, okvis::ViSlamBackend::FinalTrajectoryPosition> gtTrajectory;
  std::mutex gtTrajectoryMutex;
  if (useGtPoses) {
    auto poses = std::make_shared<GtPoses>();
    const std::string filename = path + "/gt_trajectory.csv";
    if (!loadGtPoses(filename, *poses, gtTrajectory)) {
      LOG(ERROR) << "Cannot load GT poses: " << filename;
      return EXIT_FAILURE;
    }
    gtPoses = std::move(poses);
    LOG(INFO) << "GT pose mapping: " << gtPoses->size() << " poses from " << filename;
  }

  std::shared_ptr<okvis::XDatasetReader> datasetReader;
  okvis::Duration deltaT(0.0); 
  if (isSubmapping) {
    datasetReader.reset(
        new okvis::XDatasetReader(path, deltaT, parameters, isLidar, isDepth));
  } else {
    datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, false, false));
  }
  datasetReader->setPlaybackRateControl(realtimePipelineEnabled, inputTargetFps);

  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string();
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR) << "The dbow file is not valid: " 
      << dBowVocDir << "/small_voc.yml.gz";
    return EXIT_FAILURE;
  }

  std::string savePath = args.size() == 4 ? args[3] : path;

  submapConfig.resultsDirectory = savePath;
  submapConfig.gsOutputDirectory = (std::filesystem::path(savePath) / "gs").string();
  submapConfig.gsRenderedRgbDirectory = (std::filesystem::path(submapConfig.gsOutputDirectory) / "rendered_rgb").string();
  if (std::filesystem::path(submapConfig.gsOptimParamsPath).is_relative()) {
    submapConfig.gsOptimParamsPath = (std::filesystem::absolute(seConfigFilename).parent_path() / submapConfig.gsOptimParamsPath).string();
  }
  std::filesystem::create_directories(savePath);

  LOG(INFO) << "Output root: " << submapConfig.resultsDirectory;
  LOG(INFO) << "GS output root: " << submapConfig.gsOutputDirectory;
  LOG(INFO) << "GS rendered RGB export: "
            << (submapConfig.gsSaveRenderedRgb ? "ON" : "OFF")
            << " -> " << submapConfig.gsRenderedRgbDirectory;

  LOG(INFO) << "Occupancy viewer: "
            << (occupancyViewerConfig.enable ? "ON" : "OFF")
            << " update_interval=" << occupancyViewerConfig.update_interval
            << " voxel_stride=" << occupancyViewerConfig.voxel_stride
            << " max_points=" << occupancyViewerConfig.max_points
            << " occupied_point_size=" << occupancyViewerConfig.occupied_point_size
            << " max_fps=" << occupancyViewerConfig.max_fps
            << " show_pose=" << (occupancyViewerConfig.show_pose ? "true" : "false")
            << " window=" << occupancyViewerConfig.window_name
            << " size=" << occupancyViewerConfig.window_width
            << "x" << occupancyViewerConfig.window_height;
  LOG(INFO) << "Realtime pipeline: "
            << (realtimePipelineEnabled ? "ON" : "OFF")
            << " input_target_fps=" << inputTargetFps
            << " depth_queue_drop_if_full="
            << (submapConfig.depthQueueDropIfFull ? "true" : "false");

  std::shared_ptr<okvis::ThreadedSlam> estimator(nullptr);
  estimator.reset(new okvis::ThreadedSlam(parameters, dBowVocDir, submapConfig));
  estimator->setBlocking(!realtimePipelineEnabled);
  std::function<void()> configureFrontendEstimatorCallbacks;

  std::shared_ptr<okvis::SubmappingInterface> seInterface(nullptr);
  if (isSubmapping) {
    seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters, gtPoses));

    seInterface->setGsDensifyConfig(gsDensifyConfig);
    seInterface->setOccupancyViewerConfig(occupancyViewerConfig);
    seInterface->setBlocking(!realtimePipelineEnabled);
  }

  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }
  if (useGtPoses) mode += "-gtpose";

  std::string finalTrajectoryCsvFile =
      savePath + "/okvis2-" + mode + "-final_trajectory.csv";
  estimator->setFinalTrajectoryCsvFile(finalTrajectoryCsvFile);
  estimator->setMapCsvFile(savePath+"/okvis2-" + mode + "-final_map.csv");

  std::shared_ptr<okvis::TrajectoryOutput> writer;
  writer.reset(new okvis::TrajectoryOutput(savePath+"/okvis2-" + mode + "_trajectory.csv", false, parameters.output.display_topview));

  const bool occupancyViewerEnabled = (isSubmapping && occupancyViewerConfig.enable);

  FpsSampleState inputFpsSampleState;
  std::mutex occupancyTrajectoryMutex;
  std::map<uint64_t, Eigen::Vector3f> occupancyTrajectoryById;
  std::atomic_bool occupancyTrajectoryDirty(false);
  auto updateOccupancyTrajectory =
      [&](std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated_states) {
        if (!occupancyViewerEnabled || !updated_states) {
          return;
        }
        std::lock_guard<std::mutex> lock(occupancyTrajectoryMutex);
        for (const auto& state_kv : *updated_states) {
          occupancyTrajectoryById[state_kv.first.value()] =
              state_kv.second.T_WS.r().cast<float>();
        }
        occupancyTrajectoryDirty.store(true);
      };
  auto mergeOccupancyTrajectory = [&](const std::map<uint64_t, Eigen::Vector3f>& trajectory) {
    if (!occupancyViewerEnabled || trajectory.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(occupancyTrajectoryMutex);
      for (const auto& state_kv : trajectory) {
        occupancyTrajectoryById[state_kv.first] = state_kv.second;
      }
    }
    occupancyTrajectoryDirty.store(true);
  };
  auto buildCallbackStateView =
      [&](const okvis::State& latest_state,
          std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated_states,
          okvis::State& latest_state_out,
          std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>>&
              updated_states_out) {
        latest_state_out = latest_state;
        if (!useGtPoses) {
          updated_states_out = updated_states;
          return;
        }
        auto states = std::make_shared<okvis::AlignedMap<okvis::StateId, okvis::State>>(
            *updated_states);
        std::lock_guard<std::mutex> lock(gtTrajectoryMutex);
        const auto applyGtPose = [&](okvis::State& state) {
          const uint64_t timestamp = state.timestamp.toNSec();
          state.T_WS = gtPoses->at(timestamp);
          gtTrajectory.at(timestamp).state_id = state.id.value();
        };
        applyGtPose(latest_state_out);
        for (auto& state : *states) applyGtPose(state.second);
        updated_states_out = std::move(states);
      };

  if (isSubmapping) {

    estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      okvis::State stateForCallbacks;
      std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>>
          updatedStatesForCallbacks;
      buildCallbackStateView(_1, _3, stateForCallbacks, updatedStatesForCallbacks);

      writer->processState(stateForCallbacks, _2, updatedStatesForCallbacks, _4);
      updateOccupancyTrajectory(updatedStatesForCallbacks);
      seInterface->stateUpdateCallback(stateForCallbacks, _2, updatedStatesForCallbacks);

    });

    if(submapConfig.useMap2MapFactors || submapConfig.useMap2LiveFactors) {
      seInterface->setAlignCallback(std::bind(&okvis::ThreadedSlam::addSubmapAlignmentConstraints, estimator,
                                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
    }
    if (occupancyViewerEnabled && !useGtPoses) {
      seInterface->setGsOptimizedTrajectoryCallback(
          [&](const std::map<uint64_t, Eigen::Vector3f>& trajectory) {
            mergeOccupancyTrajectory(trajectory);
          });
    }
    if (!useGtPoses) {
      seInterface->setGsOptimizedPoseCallback(
          [&](const std::map<uint64_t, Eigen::Matrix4f>& poses_by_state_id) {
            estimator->enqueueGsOptimizedPoses(poses_by_state_id);
          });
    }
  }
  else {
    configureFrontendEstimatorCallbacks = [&]() {
      estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      okvis::State stateForCallbacks;
      std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>>
          updatedStatesForCallbacks;
      buildCallbackStateView(_1, _3, stateForCallbacks, updatedStatesForCallbacks);

      writer->processState(stateForCallbacks, _2, updatedStatesForCallbacks, _4);
      updateOccupancyTrajectory(updatedStatesForCallbacks);
      });
    };
    configureFrontendEstimatorCallbacks();
  }

  datasetReader->setImuCallback(
      [&](const okvis::Time& stamp, const Eigen::Vector3d& acceleration,
          const Eigen::Vector3d& angularVelocity) {
        return estimator->addImuMeasurement(
            stamp, acceleration, angularVelocity);
      });
  datasetReader->setImagesCallback(
      [&](const okvis::Time& stamp,
          const std::map<size_t, cv::Mat>& images,
          const std::map<size_t, cv::Mat>& depthImages) {
        return estimator->addImages(stamp, images, depthImages);
      });

  if(parameters.gps){
      if((*parameters.gps).type == "cartesian"){
          datasetReader->setGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
      }
      else if((*parameters.gps).type == "geodetic" || (*parameters.gps).type == "geodetic-leica"){
          datasetReader->setGeodeticGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGeodeticGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
      }
  }

  int datasetDepthSanityLogsRemaining =
      (isSubmapping && isDepth) ? 3 : 0;
  const float datasetDepthNearPlaneM =
      std::max(0.0f, static_cast<float>(submapConfig.near_plane));
  const float datasetDepthFarPlaneM =
      static_cast<float>(submapConfig.far_plane);
  auto logDatasetDepthSanity =
      [&](const std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames) {
        if (datasetDepthSanityLogsRemaining <= 0) {
          return;
        }

        for (const auto& cam_idx_frames : frames) {
          for (const auto& cam_measurement : cam_idx_frames.second) {
            if (cam_measurement.measurement.depthImage.empty()) {
              continue;
            }

            const float decoded_median_depth =
                sampledMedianDepth(cam_measurement.measurement.depthImage);
            if (!std::isfinite(decoded_median_depth)) {
              continue;
            }

            --datasetDepthSanityLogsRemaining;
            const int sanity_index = 3 - datasetDepthSanityLogsRemaining;
            const float effective_median_depth =
                decoded_median_depth * submapConfig.depthScalingFactor;
            LOG(INFO) << "Dataset depth sanity [" << sanity_index << "/3]"
                      << ": decoded_median=" << decoded_median_depth
                      << ", depth_scaling_factor="
                      << submapConfig.depthScalingFactor
                      << ", scaled_median=" << effective_median_depth
                      << ", near_plane=" << datasetDepthNearPlaneM
                      << ", far_plane=" << datasetDepthFarPlaneM;

            const bool too_near =
                (datasetDepthNearPlaneM > 0.0f)
                && (effective_median_depth < 0.1f * datasetDepthNearPlaneM);
            const bool too_far =
                std::isfinite(datasetDepthFarPlaneM)
                && (datasetDepthFarPlaneM > 0.0f)
                && (effective_median_depth > 4.0f * datasetDepthFarPlaneM);
            if (too_near || too_far) {
              LOG(WARNING) << "Depth sanity check suggests an unexpected depth scale"
                           << " (scaled_median=" << effective_median_depth
                           << ", near_plane=" << datasetDepthNearPlaneM
                           << ", far_plane=" << datasetDepthFarPlaneM
                           << ")";
            }
            return;
          }
        }
      };

  if (isSubmapping) {
    if (isDepth) {
      datasetReader->setDepthImageCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        bool estimatorAdd = true;
        bool hasDepthMeasurement = false;
        if (submapConfig.useMap2LiveFactors) {
          for(const auto& cam_idx_frames : frames) {
            for(const auto& cam_measurement : cam_idx_frames.second) {
              if(!cam_measurement.measurement.depthImage.empty()) {
                hasDepthMeasurement = true;
                estimatorAdd = estimatorAdd
                               && estimator->addTaggedDepthMeasurement(
                                      cam_measurement);
              }
            }
          }
        }
        logDatasetDepthSanity(frames);
        const bool mapAdd = seInterface->addDepthMeasurement(frames);
        if (!hasDepthMeasurement) {
          estimatorAdd = true;
        }
        return (estimatorAdd && mapAdd);    
      });
    }

    seInterface->start();
  }

  if(!datasetReader->startStreaming()) {
      LOG(ERROR) << "Failure with datasetReader streaming.";
      return EXIT_FAILURE;
  }

  std::atomic_bool trackingWorkerStop(false);
  std::thread trackingWorker;
  const auto startTrackingWorker = [&]() {
    if (!realtimePipelineEnabled || trackingWorker.joinable()) return;
    trackingWorkerStop.store(false);
    trackingWorker = std::thread([&]() {
      while (!trackingWorkerStop.load()) {
        if (!estimator->processFrame()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    });
  };
  startTrackingWorker();

  std::unique_ptr<okvis::OkvisOccupancyViewer3D> occupancyViewer;
  if (isSubmapping && occupancyViewerConfig.enable) {
    okvis::OkvisOccupancyViewer3D::Options viewerOptions;
    viewerOptions.window_name = occupancyViewerConfig.window_name;
    viewerOptions.max_fps = occupancyViewerConfig.max_fps;
    viewerOptions.show_pose = occupancyViewerConfig.show_pose;

    viewerOptions.occupied_point_size = occupancyViewerConfig.occupied_point_size;

    viewerOptions.max_points = occupancyViewerConfig.max_points;
    viewerOptions.window_width = occupancyViewerConfig.window_width;
    viewerOptions.window_height = occupancyViewerConfig.window_height;
    occupancyViewer.reset(new okvis::OkvisOccupancyViewer3D(viewerOptions));
    occupancyViewer->start();
  }

  okvis::Time startTime = okvis::Time::now();
  bool gsWindowInit = false;
  bool gsPreviewTimingStarted = false;
  size_t gsPreviewCount = 0;
  uint64_t latestGsMappedFrameCount = 0;
  bool gsGlobalFpsFrozen = false;
  double gsPreGlobalInputFpsSum = 0.0;
  double gsPreGlobalTrackingFpsSum = 0.0;
  double gsPreGlobalSystemFpsSum = 0.0;
  double gsPreGlobalBackendFpsSum = 0.0;
  uint64_t gsPreGlobalFpsSampleCount = 0;
  float gsFrozenInputFps = 0.0f;
  float gsFrozenTrackingFps = 0.0f;
  float gsFrozenSystemFps = 0.0f;
  float gsFrozenBackendFps = 0.0f;
  std::chrono::steady_clock::time_point gsPreviewStartTime;
  auto showGsPreview = [&](okvis::GsPreviewFrame& gsPreview) {
    const auto now = std::chrono::steady_clock::now();
    const bool isGlobalOptPreview = (gsPreview.global_iter > 0);
    const uint64_t inputFrameCount = datasetReader->inputTickCount();
    if (!isGlobalOptPreview) {
      if (!gsPreviewTimingStarted) {
        gsPreviewTimingStarted = true;
        gsPreviewStartTime = now;
      }
      gsPreviewCount++;
      const double elapsed_s =
          std::chrono::duration<double>(now - gsPreviewStartTime).count();
      if (elapsed_s > 0.0) {
        gsPreview.fps = static_cast<float>(
            static_cast<double>(gsPreviewCount) / elapsed_s);
      } else {
        gsPreview.fps = 0.0f;
      }
      gsPreview.input_fps = sampleRateHz(inputFrameCount, inputFpsSampleState);
      gsPreview.tracking_fps = estimator->latestFrontendTrackingFps();

      gsPreGlobalInputFpsSum += gsPreview.input_fps;
      gsPreGlobalTrackingFpsSum += gsPreview.tracking_fps;
      gsPreGlobalSystemFpsSum += gsPreview.fps;
      gsPreGlobalBackendFpsSum += gsPreview.backend_fps;
      ++gsPreGlobalFpsSampleCount;
      gsFrozenInputFps = static_cast<float>(
          gsPreGlobalInputFpsSum / static_cast<double>(gsPreGlobalFpsSampleCount));
      gsFrozenTrackingFps = static_cast<float>(
          gsPreGlobalTrackingFpsSum / static_cast<double>(gsPreGlobalFpsSampleCount));
      gsFrozenSystemFps = static_cast<float>(
          gsPreGlobalSystemFpsSum / static_cast<double>(gsPreGlobalFpsSampleCount));
      gsFrozenBackendFps = static_cast<float>(
          gsPreGlobalBackendFpsSum / static_cast<double>(gsPreGlobalFpsSampleCount));
    } else {
      if (!gsGlobalFpsFrozen) {
        gsGlobalFpsFrozen = true;
        if (gsPreGlobalFpsSampleCount == 0) {
          gsFrozenInputFps = sampleRateHz(inputFrameCount, inputFpsSampleState);
          gsFrozenTrackingFps = estimator->latestFrontendTrackingFps();
          gsFrozenSystemFps = gsPreview.fps;
          gsFrozenBackendFps = gsPreview.backend_fps;
        }
      }
      gsPreview.input_fps = gsFrozenInputFps;
      gsPreview.tracking_fps = gsFrozenTrackingFps;
      gsPreview.fps = gsFrozenSystemFps;
      gsPreview.backend_fps = gsFrozenBackendFps;
    }
    if (gsPreview.id >= 0) {
      const uint64_t mappedFrameCandidate = static_cast<uint64_t>(gsPreview.id) + 1;
      latestGsMappedFrameCount = std::max(latestGsMappedFrameCount, mappedFrameCandidate);
    }
    uint64_t mappedFrameCountToDisplay =
        realtimePipelineEnabled ? latestGsMappedFrameCount : inputFrameCount;
    mappedFrameCountToDisplay =
        std::min(mappedFrameCountToDisplay, inputFrameCount);
    if (!gsWindowInit) {
      cv::namedWindow("GO Mapping", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
      cv::resizeWindow("GO Mapping", 1280, 720);
      gsWindowInit = true;
    }
    cv::imshow("GO Mapping",
               composeGoMappingWindow(
                   gsPreview, mappedFrameCountToDisplay, inputFrameCount));
  };
  auto submitOccupancyTrajectory = [&](const bool force_submit) -> bool {
    if (!occupancyViewer) {
      return false;
    }
    if (!force_submit) {
      if (!occupancyTrajectoryDirty.exchange(false)) {
        return false;
      }
    } else {
      occupancyTrajectoryDirty.store(false);
    }

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> trajectory;
    {
      std::lock_guard<std::mutex> lock(occupancyTrajectoryMutex);
      trajectory.reserve(occupancyTrajectoryById.size());
      for (const auto& state_kv : occupancyTrajectoryById) {
        trajectory.push_back(state_kv.second);
      }
    }
    if (!trajectory.empty()) {
      occupancyViewer->submitTrajectory(trajectory);
      return true;
    }
    return false;
  };
  bool mappingFinalized = false;
  auto pumpOccupancyViewer = [&]() -> bool {
    if (!seInterface) {
      return false;
    }

    bool fisher_mode_changed = false;
    const bool fisher_enabled = occupancyViewer && occupancyViewer->useFisherOverlayEnabled();
    if (seInterface->isGsFisherVisualizationEnabled() != fisher_enabled) {
      seInterface->setGsFisherVisualizationEnabled(fisher_enabled);
      fisher_mode_changed = true;
    }
    const bool global_fisher_enabled =
        occupancyViewer && occupancyViewer->useGlobalFisherOverlayEnabled();
    if (seInterface->isGsGlobalFisherVisualizationEnabled() != global_fisher_enabled) {
      seInterface->setGsGlobalFisherVisualizationEnabled(global_fisher_enabled);
      fisher_mode_changed = true;
    }

    if (!occupancyViewer) {
      return false;
    }

    bool submitted = false;
    okvis::OccupancyPreviewFrame occupancyFrame;
    while (seInterface->publishOccupancyPreview(occupancyFrame)) {
      occupancyViewer->submitFrame(occupancyFrame);
      submitted = true;
    }
    if (mappingFinalized && fisher_mode_changed
        && seInterface->buildOccupancyPreviewSnapshot(occupancyFrame, true)) {
      occupancyViewer->submitFrame(occupancyFrame);
      submitted = true;
    }
    submitted = submitOccupancyTrajectory(false) || submitted;
    return submitted;
  };
  auto submitFinalOccupancySnapshot = [&]() {
    if (!occupancyViewer || !seInterface) {
      return;
    }

    pumpOccupancyViewer();
    okvis::OccupancyPreviewFrame snapshotFrame;
    if (seInterface->buildOccupancyPreviewSnapshot(snapshotFrame, mappingFinalized)) {
      occupancyViewer->submitFrame(snapshotFrame);
    }

    submitOccupancyTrajectory(true);

    pumpOccupancyViewer();
  };
  auto refreshOccupancyTrajectoryFromFinalSource = [&]() -> bool {
    if (!occupancyViewerEnabled) {
      return false;
    }
    std::map<uint64_t, Eigen::Vector3f> finalTrajectoryById;
    if (useGtPoses) {
      std::lock_guard<std::mutex> lock(gtTrajectoryMutex);
      for (const auto& pose : gtTrajectory) {
        if (pose.second.state_id != 0) {
          finalTrajectoryById[pose.second.state_id] =
              Eigen::Vector3f(pose.second.px, pose.second.py, pose.second.pz);
        }
      }
    } else {
      if (!estimator->getFinalTrajectoryPositions(finalTrajectoryById)
          || finalTrajectoryById.empty()) {
        return false;
      }
    }
    {
      std::lock_guard<std::mutex> lock(occupancyTrajectoryMutex);
      occupancyTrajectoryById = std::move(finalTrajectoryById);
    }
    occupancyTrajectoryDirty.store(true);
    return true;
  };

  int progress = 0;
  bool datasetreaderFinished = false;
  std::unordered_set<std::string> openCvWindows;
  okvis::GsFinalizationSummary gsFinalSummary;
  bool haveGsFinalSummary = false;
  while(true){

      if(!datasetreaderFinished){
          if (!realtimePipelineEnabled) {
            estimator->processFrame();
          }
          std::map<std::string, cv::Mat> images;
          if (!headless) estimator->display(images);
          for(const auto & image : images) {
            cv::imshow(image.first, image.second);
            openCvWindows.insert(image.first);
          }
          cv::Mat topView;
          if (!headless) writer->drawTopView(topView);
          if(!topView.empty()) {
            cv::imshow("OKVIS 2 Top View", topView);
            openCvWindows.insert("OKVIS 2 Top View");
          }
          bool hasGsPreview = false;
          okvis::GsPreviewFrame gsPreview;
          if (isSubmapping && seInterface->publishGsPreview(gsPreview)) {
            hasGsPreview = true;
            showGsPreview(gsPreview);
            openCvWindows.insert("GO Mapping");
          }
          pumpOccupancyViewer();
          if(!images.empty() || !topView.empty() || hasGsPreview) {
            const int key = cv::waitKey(2);
            (void)key;
          } else if (realtimePipelineEnabled) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
      }

      if(!datasetReader->isStreaming()) {
          datasetreaderFinished = true;
          std::cout << "\r DatasetReader Finished!" << std::endl;

          if(datasetreaderFinished){
              if (realtimePipelineEnabled) {
                trackingWorkerStop.store(true);
                if (trackingWorker.joinable()) {
                  trackingWorker.join();
                }
              }

              estimator->stopThreading();
              if (isSubmapping) {
                while(!seInterface->finishedIntegrating()){
                  pumpOccupancyViewer();
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  LOG(INFO) << "waiting for integrator to finish!";
                }
                pumpOccupancyViewer();
              }
              estimator->writeFinalTrajectoryCsv();
              if(parameters.estimator.do_final_ba) {
                LOG(INFO) << "final full BA...";
                estimator->doFinalBa();
                finalTrajectoryCsvFile =
                    savePath + "/okvis2-" + mode + "-final-ba_trajectory.csv";
                estimator->setFinalTrajectoryCsvFile(finalTrajectoryCsvFile);
                estimator->writeFinalTrajectoryCsv();

                if (isSubmapping) {
                  while(!seInterface->finishedIntegrating()){
                    pumpOccupancyViewer();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    LOG(INFO) << "waiting for integrator to finish!";
                  }
                  pumpOccupancyViewer();
                }
              }
              refreshOccupancyTrajectoryFromFinalSource();
              if (isSubmapping && occupancyViewer) {
                submitFinalOccupancySnapshot();
              }
              LOG(INFO) <<"total processing time OKVIS only " << (okvis::Time::now() - startTime) << " s" << std::endl;

              if (isSubmapping) {
                seInterface->setFinished();
                if (submapConfig.enableGsMapping) {
                  std::vector<okvis::ViSlamBackend::FinalTrajectoryPosition>
                      finalTrajectoryPoses;
                  std::string gsFinalTrajectoryCsvPath = finalTrajectoryCsvFile;
                  if (useGtPoses) {
                    {
                      std::lock_guard<std::mutex> lock(gtTrajectoryMutex);
                      for (const auto& pose : gtTrajectory) {
                        finalTrajectoryPoses.push_back(pose.second);
                      }
                    }
                    gsFinalTrajectoryCsvPath = savePath + "/gt_trajectory.csv";
                    if (!writeTrajectoryCsv(gsFinalTrajectoryCsvPath, finalTrajectoryPoses)) {
                      LOG(ERROR) << "Cannot write " << gsFinalTrajectoryCsvPath;
                      return EXIT_FAILURE;
                    }
                  } else {
                    if (!estimator->getFinalTrajectoryPoses(finalTrajectoryPoses)
                        || finalTrajectoryPoses.empty()) {
                      LOG(WARNING) << "GS finalization: in-memory final pose provider unavailable, "
                                   << "falling back to CSV-only poses.";
                    } else {
                      LOG(INFO) << "GS finalization: loaded in-memory final poses "
                                << finalTrajectoryPoses.size();
                    }
                  }
                  std::atomic_bool gsFinalizeDone(false);
                  bool gsFinalizeResult = true;
                  std::thread gsFinalizeThread([&]() {
                    gsFinalizeResult =
                        seInterface->finalizeGsMapping(path,
                                                       gsFinalTrajectoryCsvPath,
                                                       finalTrajectoryPoses);
                    gsFinalizeDone.store(true);
                  });

                  while (!gsFinalizeDone.load()) {
                    okvis::GsPreviewFrame gsPreview;
                    if (seInterface->publishGsPreview(gsPreview)) {
                      showGsPreview(gsPreview);
                      openCvWindows.insert("GO Mapping");
                      pumpOccupancyViewer();
                      cv::waitKey(2);
                    } else {
                      pumpOccupancyViewer();
                      std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                  }
                  if (gsFinalizeThread.joinable()) {
                    gsFinalizeThread.join();
                  }
                  okvis::GsPreviewFrame gsPreview;
                  while (seInterface->publishGsPreview(gsPreview)) {
                    showGsPreview(gsPreview);
                    openCvWindows.insert("GO Mapping");
                    pumpOccupancyViewer();
                    cv::waitKey(2);
                  }
                  if (!gsFinalizeResult) {
                    LOG(WARNING) << "GS finalization reported failure.";
                  }
                  haveGsFinalSummary =
                      seInterface->getLastGsFinalizationSummary(gsFinalSummary);
                }
                if (submapConfig.write_mesh_output) {
                  seInterface->saveAllSubmapMeshes();
                }
                cv::Mat finalSubmapPlot;

              }
              estimator->saveMap(); 
              const double totalProcessingSec =
                  (okvis::Time::now() - startTime).toSec();
              std::ostringstream runtimeSummary;
              runtimeSummary << std::fixed << std::setprecision(4)
                             << "\nTotal processing time: " << totalProcessingSec << " s\n";
              if (haveGsFinalSummary) {
                const double slamWithoutGlobalOptSec =
                    std::max(0.0, totalProcessingSec
                                      - gsFinalSummary.global_optimization_time_sec);
                runtimeSummary << std::setprecision(3)
                               << "Avg. fps (pre-global-opt System): "
                               << gsFinalSummary.pre_global_opt_system_fps << "\n"
                               << std::setprecision(4)
                               << "Global opt. time: "
                               << gsFinalSummary.global_optimization_time_sec << " s\n"
                               << "Processing time (w/o global opt.): "
                               << slamWithoutGlobalOptSec << " s\n";
                if (gsFinalSummary.gpu_memory_usage_mb >= 0.0) {
                  runtimeSummary << std::setprecision(0)
                                 << "GPU memory usage: "
                                 << gsFinalSummary.gpu_memory_usage_mb << " MB\n";
                } else {
                  runtimeSummary << "GPU memory usage: unavailable\n";
                }
                runtimeSummary << "#Gaussians: "
                               << gsFinalSummary.final_global_gaussian_count << "\n"
                               << "#Keyframes: " << gsFinalSummary.global_opt_kf_count << "\n"
                               << "Saving at " << gsFinalSummary.save_iteration_frames << " frames\n";
                if (!appendGsRunSummaryCsv(savePath, mode, totalProcessingSec, gsFinalSummary)) {
                  LOG(WARNING) << "Failed to write GS run summary CSV to "
                               << (boost::filesystem::path(savePath)
                                   / ("okvis2-" + mode + "-run_summary.csv"))
                                      .string();
                }
              }
              std::cout << runtimeSummary.str() << std::flush;
              break;
          }
      }

      int newProgress = int(datasetReader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
      if (newProgress>progress) {
          LOG(INFO) << okvis::timing::Timing::print();
      }
#endif
      if (newProgress>progress) {
          progress = newProgress;
          std::cout << "\rProgress: "
                    << progress << "% "
                    << std::flush;
      }
  }

  if (trackingWorker.joinable()) {
    trackingWorkerStop.store(true);
    trackingWorker.join();
  }

  mappingFinalized = true;
  submitFinalOccupancySnapshot();

  std::cout << "Processing complete. Output: " << savePath << std::endl;

  if (!openCvWindows.empty() || (occupancyViewer && occupancyViewer->running())) {
    std::cout << "Close the viewer windows manually, or press Ctrl+C to exit."
              << std::endl;
    while (true) {
      pumpOccupancyViewer();
      for (auto it = openCvWindows.begin(); it != openCvWindows.end();) {
        if (!windowVisible(*it)) {
          it = openCvWindows.erase(it);
        } else {
          ++it;
        }
      }
      const bool hasOpenCvWindow = !openCvWindows.empty();
      const bool hasOccupancyWindow = occupancyViewer && occupancyViewer->running();
      if (!hasOpenCvWindow && !hasOccupancyWindow) {
        break;
      }

      if (hasOpenCvWindow) {
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') {
          for (const auto& window_name : openCvWindows) {
            cv::destroyWindow(window_name);
          }
          openCvWindows.clear();
          break;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
    }
  }

  if (occupancyViewer) {
    occupancyViewer->stop();
  }
  return EXIT_SUCCESS;
}
