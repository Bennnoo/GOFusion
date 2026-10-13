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
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_PARAMETERS_HPP_
#define INCLUDE_OKVIS_PARAMETERS_HPP_

#include <memory>
#include <set>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <okvis/MappingImagePreprocessor.hpp>
#include <okvis/Time.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <optional>

namespace okvis {

struct CameraCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::kinematics::Transformation T_SC;   
  Eigen::Vector2i imageDimension;           
  Eigen::VectorXd distortionCoefficients;   
  Eigen::Vector2d focalLength;              
  Eigen::Vector2d principalPoint;           
  std::string cameraModel;                  
  Eigen::Vector2d eucmParameters;           

  std::string distortionType;

  cameras::NCameraSystem::CameraType cameraType; 
};

struct CameraParameters{
  double timestamp_tolerance; 
  std::set<size_t> sync_cameras; 

  double image_delay;

  struct OnlineCalibrationParameters {
    bool do_extrinsics; 
    bool do_extrinsics_final_ba; 
    double sigma_r; 
    double sigma_alpha; 
    double sigma_r_final_ba; 
    double sigma_alpha_final_ba; 
  };

  OnlineCalibrationParameters online_calibration; 
};

struct ImuParameters{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool use; 
  okvis::kinematics::Transformation T_BS; 
  double a_max;  
  double g_max;  
  double sigma_g_c;  
  double sigma_bg;  
  double sigma_a_c;  
  double sigma_ba;  
  double sigma_gw_c; 
  double sigma_aw_c; 
  Eigen::Vector3d g0;  
  Eigen::Vector3d a0;  
  double g;  
  Eigen::Vector3d s_a; 
};

struct FrontendParameters {
  double detection_threshold; 
  double absolute_threshold; 
  double matching_threshold; 
  int octaves; 
  int max_num_keypoints; 
  double keyframe_overlap; 

  bool parallelise_detection; 
  int num_matching_threads; 

  size_t tracking_image_resolution_downsampling = 1;
};

struct EstimatorParameters {
  int num_keyframes; 
  int num_loop_closure_frames; 
  int num_imu_frames; 
  bool do_loop_closures; 
  bool do_final_ba; 
  bool enforce_realtime; 
  int realtime_min_iterations; 
  int realtime_max_iterations; 
  double realtime_time_limit; 
  int realtime_num_threads; 
  int full_graph_iterations; 
  int full_graph_num_threads; 
  double p_dbow; 
  double drift_percentage_heuristic; 
};

struct OutputParameters {
    bool display_topview; 
    bool display_matches; 
    bool display_overhead; 
    bool enable_submapping; 
};

struct GpsParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::string type; 
    Eigen::Vector3d r_SA; 
    double yawErrorThreshold; 
    bool robustGpsInit; 

    GpsParameters() : type("none"), r_SA(Eigen::Vector3d(0., 0., 0.)),
                      yawErrorThreshold(0.), robustGpsInit(false)
                      {}
};

struct LidarParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SL;   
    float elevation_resolution_angle; 
    float azimuth_resolution_angle; 
};

struct ViParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::cameras::NCameraSystem nCameraSystem;  

  std::shared_ptr<const okvis::cameras::NCameraSystem>
      effectiveTrackingCameraSystem;
  CameraParameters camera; 
  ImuParameters imu; 
  std::optional<GpsParameters> gps; 
  std::optional<LidarParameters> lidar; 
  FrontendParameters frontend; 
  EstimatorParameters estimator; 
  OutputParameters output; 
  CameraCalibration rgb;  

  std::shared_ptr<const MappingImagePreprocessor> mappingInputPreprocessor;

  size_t mappingInputDownsampling = 1;

  const okvis::cameras::NCameraSystem& trackingCameraSystem() const {
    return effectiveTrackingCameraSystem
               ? *effectiveTrackingCameraSystem
               : nCameraSystem;
  }

  bool mappingInputEnabledFor(const size_t cameraIndex) const {
    return mappingInputPreprocessor && mappingInputPreprocessor->enabled()
           && mappingInputPreprocessor->targetCameraIndex() == cameraIndex;
  }

  std::shared_ptr<const cameras::CameraBase> mappingCameraGeometry(
      const size_t cameraIndex,
      const size_t downsampling = 1) const {
    if (mappingInputEnabledFor(cameraIndex)) {
      return mappingInputPreprocessor->targetGeometry(downsampling);
    }
    return nCameraSystem.cameraGeometry(cameraIndex);
  }

  std::shared_ptr<const kinematics::Transformation> mappingT_SC(
      const size_t cameraIndex) const {
    if (mappingInputEnabledFor(cameraIndex)) {
      return mappingInputPreprocessor->targetT_SC();
    }
    return nCameraSystem.T_SC(cameraIndex);
  }
};

} 

#endif 
