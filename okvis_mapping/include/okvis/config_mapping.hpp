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

#ifndef INCLUDE_OKVIS_LIDARMAPPING_CONFIG_HPP
#define INCLUDE_OKVIS_LIDARMAPPING_CONFIG_HPP

#include <se/map/map.hpp>
#include <se/sensor/sensor.hpp>
#include <se/sensor/lidar.hpp>
#include <stdexcept>

namespace se {

struct SubMapConfig{

    std::string resultsDirectory;

    bool write_mesh_output;

    size_t sensorMeasurementDownsampling;

    size_t depthImageResDownsampling;

    size_t depthQueueSize = 4;

    bool depthQueueDropIfFull = false;

    bool realtimePipelineEnabled = false;

    double inputTargetFps = 60.0;

    size_t cameraInputQueueSize = 0;

    size_t depthMeasurementQueueSize = 0;

    size_t nonBlockingStateUpdateQueueSize = 0;

    size_t nonBlockingSupereightQueueSize = 0;

    size_t submapKfThreshold = 5;

    float submapOverlapRatio = 0.4;

    size_t submapMinFrames = 1;

    bool useMap2MapFactors = false;

    bool useMap2LiveFactors = false;

    size_t numSubmapFactors = 200;

    float voxelGridResolution = 0.1f;

    float sensorError = 0.01f;


    float depthScalingFactor = 1.0f;

    bool enableGsMapping = false;

    bool gsShowPreview = false;

    bool gsRunGlobalOptimization = true;

    std::string gsOptimParamsPath;

    std::string gsOutputDirectory;

    bool gsSaveRenderedRgb = false;

    bool gsSaveFisher = true;

    int gsKeepRecentFrozenSubmapsGpu = 3;

    bool gsOptimizeCudaMemory = true;

    std::string gsRenderedRgbDirectory;

    float near_plane = 0.1;

    float far_plane = 5.0;

    SubMapConfig() : resultsDirectory("/home/"), write_mesh_output(false),
      sensorMeasurementDownsampling(1), depthImageResDownsampling(1){};

    SubMapConfig(const std::string& yaml_file) : SubMapConfig()
    {

      cv::FileStorage fs;
      try {
        if (!fs.open(yaml_file, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML)) {
          std::cerr << "Error: couldn't read configuration file " << yaml_file << "\n";
          return;
        }
      }
      catch (const cv::Exception& e) {

        std::cerr << "Error: invalid YAML in configuration file " << yaml_file << "\n";
        return;
      }

      const cv::FileNode node = fs["general"];
      if (node.type() != cv::FileNode::MAP) {
        std::cerr << "Warning: using default map configuration, no \"map\" section found in "
                  << yaml_file << "\n";
        return;
      }

      se::yaml::subnode_as_string(node, "results_directory", resultsDirectory);
      se::yaml::subnode_as_bool(node, "write_mesh_output", write_mesh_output);
      se::yaml::subnode_as_size_t(node, "sensor_measurement_downsampling", sensorMeasurementDownsampling);
      int depth_downsampling = static_cast<int>(depthImageResDownsampling);
      se::yaml::subnode_as_int(
          node, "depth_image_resolution_downsampling", depth_downsampling);
      if (depth_downsampling < 1) {
        throw std::invalid_argument(
            "depth_image_resolution_downsampling must be at least 1 in " + yaml_file);
      }
      depthImageResDownsampling = static_cast<size_t>(depth_downsampling);
      se::yaml::subnode_as_size_t(node, "depth_queue_size", depthQueueSize);
      se::yaml::subnode_as_bool(node, "depth_queue_drop_if_full", depthQueueDropIfFull);
      se::yaml::subnode_as_bool(node, "realtime_pipeline_enabled", realtimePipelineEnabled);
      float inputTargetFpsRead = static_cast<float>(inputTargetFps);
      se::yaml::subnode_as_float(node, "input_target_fps", inputTargetFpsRead);
      inputTargetFps = static_cast<double>(inputTargetFpsRead);
      se::yaml::subnode_as_size_t(node, "camera_input_queue_size", cameraInputQueueSize);
      se::yaml::subnode_as_size_t(node, "depth_measurement_queue_size", depthMeasurementQueueSize);
      se::yaml::subnode_as_size_t(node, "non_blocking_state_update_queue_size", nonBlockingStateUpdateQueueSize);
      se::yaml::subnode_as_size_t(node, "non_blocking_supereight_queue_size", nonBlockingSupereightQueueSize);
      se::yaml::subnode_as_size_t(node, "submap_kf_threshold", submapKfThreshold);
      se::yaml::subnode_as_float(node, "submap_overlap_ratio", submapOverlapRatio);
      se::yaml::subnode_as_size_t(node, "submap_min_frames", submapMinFrames);
      se::yaml::subnode_as_bool(node, "use_map_to_map_factors", useMap2MapFactors);
      se::yaml::subnode_as_bool(node, "use_map_to_live_factors", useMap2LiveFactors);
      se::yaml::subnode_as_size_t(node, "n_factors_per_state", numSubmapFactors);
      se::yaml::subnode_as_float(node, "voxel_grid_resolution", voxelGridResolution);
      se::yaml::subnode_as_float(node, "sensor_error", sensorError);
      se::yaml::subnode_as_float(node, "depth_scaling_factor", depthScalingFactor);

      se::yaml::subnode_as_bool(node, "enable_gs_mapping", enableGsMapping);
      se::yaml::subnode_as_bool(node, "gs_show_preview", gsShowPreview);

      se::yaml::subnode_as_bool(node, "gs_run_global_optimization", gsRunGlobalOptimization);
      se::yaml::subnode_as_string(node, "gs_optim_params_path", gsOptimParamsPath);
      se::yaml::subnode_as_string(node, "gs_output_directory", gsOutputDirectory);
      se::yaml::subnode_as_bool(node, "gs_save_rendered_rgb", gsSaveRenderedRgb);
      se::yaml::subnode_as_bool(node, "gs_save_fisher", gsSaveFisher);
      se::yaml::subnode_as_int(node, "gs_keep_recent_frozen_submaps_gpu", gsKeepRecentFrozenSubmapsGpu);
      se::yaml::subnode_as_bool(node, "gs_optimize_cuda_memory", gsOptimizeCudaMemory);
      se::yaml::subnode_as_string(node, "gs_rendered_rgb_directory", gsRenderedRgbDirectory);

      se::yaml::subnode_as_float(node, "far_plane", far_plane);
      se::yaml::subnode_as_float(node, "near_plane", near_plane);
    }

    friend std::ostream& operator<<(std::ostream& os, const SubMapConfig& c)
    {
      os << str_utils::value_to_pretty_str(c.resultsDirectory, "results_directory") << " \n";
      os << str_utils::bool_to_pretty_str(c.write_mesh_output, "write_mesh_output") << " \n";
      os << str_utils::value_to_pretty_str(c.sensorMeasurementDownsampling, "sensor_measurement_downsampling") << " \n";
      os << str_utils::value_to_pretty_str(c.depthImageResDownsampling, "depth_image_res_downsampling") << " \n";
      os << str_utils::value_to_pretty_str(c.depthQueueSize, "depth_queue_size") << " \n";
      os << str_utils::bool_to_pretty_str(c.depthQueueDropIfFull, "depth_queue_drop_if_full") << " \n";
      os << str_utils::bool_to_pretty_str(c.realtimePipelineEnabled, "realtime_pipeline_enabled") << " \n";
      os << str_utils::value_to_pretty_str(c.inputTargetFps, "input_target_fps") << " \n";
      os << str_utils::value_to_pretty_str(c.cameraInputQueueSize, "camera_input_queue_size") << " \n";
      os << str_utils::value_to_pretty_str(c.depthMeasurementQueueSize, "depth_measurement_queue_size") << " \n";
      os << str_utils::value_to_pretty_str(c.nonBlockingStateUpdateQueueSize, "non_blocking_state_update_queue_size") << " \n";
      os << str_utils::value_to_pretty_str(c.nonBlockingSupereightQueueSize, "non_blocking_supereight_queue_size") << " \n";
      os << str_utils::value_to_pretty_str(c.submapKfThreshold, "submap_kf_threshold") << " \n";
      os << str_utils::value_to_pretty_str(c.submapOverlapRatio, "submap_overlap_ratio") << " \n";
      os << str_utils::value_to_pretty_str(c.submapMinFrames, "submap_min_frames") << " \n";
      os << str_utils::bool_to_pretty_str(c.useMap2MapFactors, "use_map_to_map_factors") << " \n";
      os << str_utils::bool_to_pretty_str(c.useMap2LiveFactors, "use_map_to_live_factors") << " \n";
      os << str_utils::value_to_pretty_str(c.numSubmapFactors, "n_factors_per_state") << " \n";
      os << str_utils::value_to_pretty_str(c.voxelGridResolution, "voxel_grid_resolution") << " \n";
      os << str_utils::value_to_pretty_str(c.sensorError, "sensor_error") << " \n";
      os << str_utils::value_to_pretty_str(c.depthScalingFactor, "depth_scaling_factor") << " \n";

      os << str_utils::bool_to_pretty_str(c.enableGsMapping, "enable_gs_mapping") << " \n";
      os << str_utils::bool_to_pretty_str(c.gsShowPreview, "gs_show_preview") << " \n";

      os << str_utils::bool_to_pretty_str(c.gsRunGlobalOptimization, "gs_run_global_optimization") << " \n";
      os << str_utils::value_to_pretty_str(c.gsOptimParamsPath, "gs_optim_params_path") << " \n";
      os << str_utils::value_to_pretty_str(c.gsOutputDirectory, "gs_output_directory") << " \n";
      os << str_utils::bool_to_pretty_str(c.gsSaveRenderedRgb, "gs_save_rendered_rgb") << " \n";
      os << str_utils::bool_to_pretty_str(c.gsSaveFisher, "gs_save_fisher") << " \n";
      os << str_utils::value_to_pretty_str(c.gsKeepRecentFrozenSubmapsGpu, "gs_keep_recent_frozen_submaps_gpu") << " \n";
      os << str_utils::bool_to_pretty_str(c.gsOptimizeCudaMemory, "gs_optimize_cuda_memory") << " \n";
      os << str_utils::value_to_pretty_str(c.gsRenderedRgbDirectory, "gs_rendered_rgb_directory") << " \n";

      os << str_utils::value_to_pretty_str(c.far_plane, "far_plane") << " \n";
      os << str_utils::value_to_pretty_str(c.near_plane, "near_plane") << " \n";
      return os;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

template<typename MapT, typename SensorT>
struct Config {
    struct MapT::Config map;
    struct MapT::DataType::Config data;
    struct SensorT::Config sensor;

    Config();

    Config(const std::string& yaml_file);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const Config<MapT, SensorT>& c);

template<typename MapT, typename SensorT>
struct OkvisSubmapsConfig {
    SubMapConfig general;
    struct Config<MapT, SensorT> se_config;

    OkvisSubmapsConfig();

    OkvisSubmapsConfig(const std::string& yaml_file);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const OkvisSubmapsConfig<MapT, SensorT>& c);

} 

#include "okvis/impl/config_mapping_impl.hpp"

#endif 
