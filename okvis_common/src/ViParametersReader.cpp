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
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

/**
 * @file ViParametersReader.cpp
 * @brief Source file for the VioParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <glog/logging.h>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/EucmCamera.hpp>
#include <okvis/MappingImagePreprocessor.hpp>

#include <opencv2/core/core.hpp>

#include <okvis/ViParametersReader.hpp>

namespace okvis {

namespace {

using NoDistortionCamera = cameras::PinholeCamera<cameras::NoDistortion>;

struct CameraBuildResult {
  std::shared_ptr<const cameras::CameraBase> geometry;
  cameras::NCameraSystem::DistortionType distortionType;
  const char* description;
};

template<typename Calibration>
Calibration scaledTrackingCalibration(
    const Calibration& source, const size_t factor) {
  if (factor == 0) {
    throw std::runtime_error(
        "tracking_image_resolution_downsampling must be a positive integer");
  }
  if (factor == 1) {
    return source;
  }

  const int source_width = source.imageDimension.x();
  const int source_height = source.imageDimension.y();
  if (source_width <= 0 || source_height <= 0) {
    throw std::runtime_error("tracking source camera has invalid dimensions");
  }
  const int target_width = source_width / static_cast<int>(factor);
  const int target_height = source_height / static_cast<int>(factor);
  if (target_width <= 0 || target_height <= 0) {
    throw std::runtime_error(
        "tracking_image_resolution_downsampling makes a camera image empty");
  }

  const double scale_x = static_cast<double>(target_width) / source_width;
  const double scale_y = static_cast<double>(target_height) / source_height;
  Calibration scaled = source;
  scaled.imageDimension = Eigen::Vector2i(target_width, target_height);
  scaled.focalLength.x() = scale_x * source.focalLength.x();
  scaled.focalLength.y() = scale_y * source.focalLength.y();
  scaled.principalPoint.x() =
      (source.principalPoint.x() + 0.5) * scale_x - 0.5;
  scaled.principalPoint.y() =
      (source.principalPoint.y() + 0.5) * scale_y - 0.5;
  return scaled;
}

template<typename Calibration>
CameraBuildResult makeConfiguredCamera(
    const Calibration& calibration,
    const bool initialise_undistort_maps,
    const bool initialise_awareness_maps) {
  const int width = calibration.imageDimension.x();
  const int height = calibration.imageDimension.y();
  if (width <= 0 || height <= 0
      || !calibration.focalLength.allFinite()
      || !calibration.principalPoint.allFinite()
      || calibration.focalLength.x() <= 0.0
      || calibration.focalLength.y() <= 0.0) {
    throw std::runtime_error(
        "camera calibration has invalid dimensions/intrinsics");
  }

  if (calibration.cameraModel == "eucm") {
    auto camera = std::make_shared<cameras::EucmCamera>(
        width, height, calibration.focalLength.x(),
        calibration.focalLength.y(), calibration.principalPoint.x(),
        calibration.principalPoint.y(), calibration.eucmParameters.x(),
        calibration.eucmParameters.y());
    if (initialise_awareness_maps) {
      camera->initialiseCameraAwarenessMaps();
    }
    return {camera, cameras::NCameraSystem::NoDistortion, "EUCM camera"};
  }

  auto require_coefficients = [&](const Eigen::Index expected) {
    if (calibration.distortionCoefficients.size() != expected) {
      throw std::runtime_error(
          "camera distortion coefficient count does not match its model");
    }
  };

  if (calibration.distortionType == "equidistant") {
    require_coefficients(4);
    auto camera = std::make_shared<cameras::PinholeCamera<
        cameras::EquidistantDistortion>>(
        width, height, calibration.focalLength.x(),
        calibration.focalLength.y(), calibration.principalPoint.x(),
        calibration.principalPoint.y(), cameras::EquidistantDistortion(
            calibration.distortionCoefficients[0],
            calibration.distortionCoefficients[1],
            calibration.distortionCoefficients[2],
            calibration.distortionCoefficients[3]));
    if (initialise_undistort_maps) {
      camera->initialiseUndistortMaps();
    }
    if (initialise_awareness_maps) {
      camera->initialiseCameraAwarenessMaps();
    }
    return {camera, cameras::NCameraSystem::Equidistant,
            "Equidistant pinhole camera"};
  }

  if (calibration.distortionType == "radialtangential"
      || calibration.distortionType == "plumb_bob") {
    require_coefficients(4);
    auto camera = std::make_shared<cameras::PinholeCamera<
        cameras::RadialTangentialDistortion>>(
        width, height, calibration.focalLength.x(),
        calibration.focalLength.y(), calibration.principalPoint.x(),
        calibration.principalPoint.y(), cameras::RadialTangentialDistortion(
            calibration.distortionCoefficients[0],
            calibration.distortionCoefficients[1],
            calibration.distortionCoefficients[2],
            calibration.distortionCoefficients[3]));
    if (initialise_undistort_maps) {
      camera->initialiseUndistortMaps();
    }
    if (initialise_awareness_maps) {
      camera->initialiseCameraAwarenessMaps();
    }
    return {camera, cameras::NCameraSystem::RadialTangential,
            "Radial tangential pinhole camera"};
  }

  if (calibration.distortionType == "radialtangential8"
      || calibration.distortionType == "plumb_bob8") {
    require_coefficients(8);
    auto camera = std::make_shared<cameras::PinholeCamera<
        cameras::RadialTangentialDistortion8>>(
        width, height, calibration.focalLength.x(),
        calibration.focalLength.y(), calibration.principalPoint.x(),
        calibration.principalPoint.y(), cameras::RadialTangentialDistortion8(
            calibration.distortionCoefficients[0],
            calibration.distortionCoefficients[1],
            calibration.distortionCoefficients[2],
            calibration.distortionCoefficients[3],
            calibration.distortionCoefficients[4],
            calibration.distortionCoefficients[5],
            calibration.distortionCoefficients[6],
            calibration.distortionCoefficients[7]));
    if (initialise_undistort_maps) {
      camera->initialiseUndistortMaps();
    }
    if (initialise_awareness_maps) {
      camera->initialiseCameraAwarenessMaps();
    }
    return {camera, cameras::NCameraSystem::RadialTangential8,
            "Radial tangential 8 pinhole camera"};
  }

  throw std::runtime_error("unrecognized distortion type "
                           + calibration.distortionType);
}

std::shared_ptr<const cameras::CameraBase> makeNoDistortionCamera(
    const cameras::CameraBase& source) {
  Eigen::VectorXd intrinsics;
  source.getIntrinsics(intrinsics);
  if (intrinsics.size() < 4) {
    throw std::runtime_error("mapping input camera has fewer than four intrinsics");
  }
  return std::make_shared<NoDistortionCamera>(
      source.imageWidth(), source.imageHeight(),
      intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
      cameras::NoDistortion());
}

Eigen::Vector2i readVector2i(const cv::FileNode& node, const char* name) {
  const cv::FileNode value = node[name];
  if (!value.isSeq() || value.size() != 2) {
    throw std::runtime_error(std::string("mapping input source missing ") + name);
  }
  return Eigen::Vector2i(static_cast<int>(value[0]), static_cast<int>(value[1]));
}

Eigen::Vector2d readVector2d(const cv::FileNode& node, const char* name) {
  const cv::FileNode value = node[name];
  if (!value.isSeq() || value.size() != 2) {
    throw std::runtime_error(std::string("mapping input source missing ") + name);
  }
  return Eigen::Vector2d(static_cast<double>(value[0]),
                         static_cast<double>(value[1]));
}

std::shared_ptr<const cameras::CameraBase> makeExplicitMappingSourceCamera(
    const cv::FileNode& source,
    const MappingImagePreprocessor::Geometry geometry) {
  const Eigen::Vector2i dimensions = readVector2i(source, "image_dimension");
  const Eigen::Vector2d focal = readVector2d(source, "focal_length");
  const Eigen::Vector2d principal = readVector2d(source, "principal_point");
  if (dimensions.x() <= 0 || dimensions.y() <= 0
      || focal.x() == 0.0 || focal.y() == 0.0) {
    throw std::runtime_error("mapping input source has invalid dimensions/intrinsics");
  }

  if (geometry == MappingImagePreprocessor::Geometry::Pinhole) {
    return std::make_shared<NoDistortionCamera>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(), cameras::NoDistortion());
  }

  std::string camera_model = "pinhole";
  if (source["cam_model"].isString()) {
    camera_model = static_cast<std::string>(source["cam_model"]);
  }
  if (camera_model == "eucm") {
    const cv::FileNode eucm = source["eucm_parameters"];
    if (!eucm.isSeq() || eucm.size() != 2) {
      throw std::runtime_error("mapping EUCM source requires eucm_parameters[2]");
    }
    return std::make_shared<cameras::EucmCamera>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(), static_cast<double>(eucm[0]),
        static_cast<double>(eucm[1]));
  }
  if (camera_model != "pinhole") {
    throw std::runtime_error("unsupported mapping input camera model: " + camera_model);
  }

  if (!source["distortion_type"].isString()) {
    throw std::runtime_error("distorted mapping source requires distortion_type");
  }
  std::string distortion_type = static_cast<std::string>(source["distortion_type"]);
  std::transform(distortion_type.begin(), distortion_type.end(),
                 distortion_type.begin(), ::tolower);
  if (distortion_type == "rational_polynomial") {
    distortion_type = "radialtangential8";
  } else if (distortion_type == "plumb_bob") {
    distortion_type = "radialtangential";
  } else if (distortion_type == "plumb_bob8") {
    distortion_type = "radialtangential8";
  }
  const cv::FileNode coefficients = source["distortion_coefficients"];
  if (!coefficients.isSeq()) {
    throw std::runtime_error(
        "distorted mapping source requires distortion_coefficients");
  }
  auto coefficient = [&](const size_t index) -> double {
    if (index >= coefficients.size()) {
      throw std::runtime_error("mapping source has too few distortion coefficients");
    }
    return static_cast<double>(coefficients[index]);
  };

  if (distortion_type == "equidistant") {
    if (coefficients.size() != 4) {
      throw std::runtime_error(
          "mapping equidistant source requires exactly 4 distortion coefficients");
    }
    return std::make_shared<
        cameras::PinholeCamera<cameras::EquidistantDistortion>>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(),
        cameras::EquidistantDistortion(
            coefficient(0), coefficient(1), coefficient(2), coefficient(3)));
  }
  if (distortion_type == "radialtangential") {
    if (coefficients.size() != 4) {
      throw std::runtime_error(
          "mapping radialtangential source requires exactly 4 distortion coefficients");
    }
    return std::make_shared<
        cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(),
        cameras::RadialTangentialDistortion(
            coefficient(0), coefficient(1), coefficient(2), coefficient(3)));
  }
  if (distortion_type == "radialtangential8") {
    if (coefficients.size() != 8) {
      throw std::runtime_error(
          "mapping radialtangential8 source requires exactly 8 distortion coefficients");
    }
    return std::make_shared<
        cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(),
        cameras::RadialTangentialDistortion8(
            coefficient(0), coefficient(1), coefficient(2), coefficient(3),
            coefficient(4), coefficient(5), coefficient(6), coefficient(7)));
  }
  if (distortion_type == "none" || distortion_type == "no_distortion") {
    for (const auto& value : coefficients) {
      if (std::abs(static_cast<double>(value)) > 1.0e-12) {
        throw std::runtime_error(
            "mapping no-distortion source contains non-zero coefficients");
      }
    }
    return std::make_shared<NoDistortionCamera>(
        dimensions.x(), dimensions.y(), focal.x(), focal.y(),
        principal.x(), principal.y(), cameras::NoDistortion());
  }
  throw std::runtime_error("unsupported mapping input distortion type: "
                           + distortion_type);
}

}  

ViParametersReader::ViParametersReader()
    : readConfigFile_(false) {
}

ViParametersReader::ViParametersReader(const std::string& filename) {

  readConfigFile(filename);
}

void ViParametersReader::readConfigFile(const std::string& filename) {

  cv::FileStorage file(filename, cv::FileStorage::READ);

  OKVIS_ASSERT_TRUE(Exception, file.isOpened(),
                    "Could not open config file: " << filename)
  LOG(INFO) << "Opened configuration file: " << filename;

  std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if(!getCameraCalibration(calibrations, file)) {
    LOG(FATAL) << "Did not find any calibration!";
  }

  int tracking_image_downsampling = 1;
  const cv::FileNode frontend_node = file["frontend_parameters"];
  if (!frontend_node["tracking_image_resolution_downsampling"].empty()) {
    parseEntry(frontend_node, "tracking_image_resolution_downsampling",
               tracking_image_downsampling);
  }
  OKVIS_ASSERT_TRUE(
      Exception, tracking_image_downsampling > 0,
      "frontend_parameters.tracking_image_resolution_downsampling must be a "
      "positive integer")
  viParameters_.frontend.tracking_image_resolution_downsampling =
      static_cast<size_t>(tracking_image_downsampling);

  const size_t tracking_factor =
      viParameters_.frontend.tracking_image_resolution_downsampling;
  const bool build_source_frontend_maps = tracking_factor == 1;
  std::shared_ptr<cameras::NCameraSystem> effective_tracking_system;
  if (tracking_factor > 1) {
    effective_tracking_system = std::make_shared<cameras::NCameraSystem>();
  }

  for (size_t camera_index = 0; camera_index < calibrations.size();
       ++camera_index) {
    const auto T_SC = std::make_shared<const kinematics::Transformation>(
        calibrations[camera_index].T_SC.r(),
        calibrations[camera_index].T_SC.q().normalized());
    const CameraBuildResult source_camera = makeConfiguredCamera(
        calibrations[camera_index], build_source_frontend_maps,
        build_source_frontend_maps);
    viParameters_.nCameraSystem.addCamera(
        T_SC, source_camera.geometry, source_camera.distortionType,
        build_source_frontend_maps, calibrations[camera_index].cameraType);

    std::stringstream transform;
    transform << calibrations[camera_index].T_SC.T();
    LOG(INFO) << source_camera.description << " " << camera_index
              << " with T_SC=\n" << transform.str();

    if (effective_tracking_system) {
      if (calibrations[camera_index].cameraType.isUsed
          && calibrations[camera_index].cameraType.depthType.createDepth) {
        OKVIS_THROW(
            Exception,
            "tracking-only image downsampling does not support a camera with "
            "slam_use: okvis-depth; use a separate mapping/depth camera entry")
      }
      const CameraCalibration effective_calibration =
          scaledTrackingCalibration(calibrations[camera_index], tracking_factor);
      const CameraBuildResult effective_camera =
          makeConfiguredCamera(effective_calibration, false, true);
      effective_tracking_system->addCamera(
          T_SC, effective_camera.geometry, effective_camera.distortionType,
          false, calibrations[camera_index].cameraType);
      LOG(INFO) << "Tracking camera " << camera_index << " downsampled "
                << calibrations[camera_index].imageDimension.x() << "x"
                << calibrations[camera_index].imageDimension.y() << " -> "
                << effective_calibration.imageDimension.x() << "x"
                << effective_calibration.imageDimension.y()
                << " (factor " << tracking_factor << ")";
    }
  }

  if (effective_tracking_system) {

    effective_tracking_system->computeOverlaps();
    viParameters_.effectiveTrackingCameraSystem = effective_tracking_system;
  }

  MappingImagePreprocessor::Config mapping_input;
  mapping_input.rgb.stream = "rgb0";
  mapping_input.depth.stream = "depth0";

  const cv::FileNode gs_mapping = file["gs_mapping_parameters"];
  if (gs_mapping.isMap()) {
    if (gs_mapping["rgb_stream"].isString()) {
      mapping_input.rgb.stream = static_cast<std::string>(gs_mapping["rgb_stream"]);
    }
    if (gs_mapping["depth_stream"].isString()) {
      mapping_input.depth.stream = static_cast<std::string>(gs_mapping["depth_stream"]);
    }
  }

  const cv::FileNode mapping_node = file["mapping_input_parameters"];
  if (!mapping_node.empty()) {
    OKVIS_ASSERT_TRUE(Exception, mapping_node.isMap(),
                      "mapping_input_parameters must be a map")
    if (!mapping_node["enabled"].empty()) {
      parseEntry(mapping_node, "enabled", mapping_input.enabled);
    }

    const cv::FileNode target_node = mapping_node["target"];
    if (mapping_input.enabled) {
      OKVIS_ASSERT_TRUE(Exception, target_node.isMap()
                                    && target_node["camera_index"].isInt(),
                        "enabled mapping_input_parameters requires target.camera_index")
      const int target_index = static_cast<int>(target_node["camera_index"]);
      OKVIS_ASSERT_TRUE(Exception, target_index >= 0
                                    && static_cast<size_t>(target_index)
                                           < viParameters_.nCameraSystem.numCameras(),
                        "mapping target camera index is out of range")
      mapping_input.targetCameraIndex = static_cast<size_t>(target_index);
      OKVIS_ASSERT_TRUE(
          Exception,
          viParameters_.nCameraSystem.cameraType(mapping_input.targetCameraIndex)
              .isUsedMapping,
          "mapping target camera must have mapping: true")
      OKVIS_ASSERT_TRUE(
          Exception,
          calibrations[mapping_input.targetCameraIndex].cameraModel == "pinhole",
          "mapping input target camera must use cam_model: pinhole; "
          "non-pinhole source models are supported, but the dense target is "
          "an undistorted pinhole grid")
      OKVIS_ASSERT_TRUE(
          Exception,
          calibrations[mapping_input.targetCameraIndex]
                  .distortionCoefficients.cwiseAbs().maxCoeff()
              <= 1.0e-12,
          "mapping input target camera must have zero distortion coefficients; "
          "put the distorted calibration under rgb/depth.source instead")
      mapping_input.targetGeometry = makeNoDistortionCamera(
          *viParameters_.nCameraSystem.cameraGeometry(mapping_input.targetCameraIndex));
      mapping_input.targetT_SC =
          viParameters_.nCameraSystem.T_SC(mapping_input.targetCameraIndex);
    }

    auto parse_modality = [&](const char* name,
                              MappingImagePreprocessor::ModalityConfig& modality) {
      const cv::FileNode node = mapping_node[name];
      if (node.empty()) return;
      OKVIS_ASSERT_TRUE(Exception, node.isMap(),
                        "mapping input modality " << name << " must be a map")
      if (node["stream"].isString()) {
        modality.stream = static_cast<std::string>(node["stream"]);
      }
      if (node["geometry"].isString()) {
        modality.geometry = MappingImagePreprocessor::geometryFromString(
            static_cast<std::string>(node["geometry"]));
      }
      if (!mapping_input.enabled
          || modality.geometry == MappingImagePreprocessor::Geometry::Target
          || modality.geometry == MappingImagePreprocessor::Geometry::SameAsRgb
          || modality.geometry == MappingImagePreprocessor::Geometry::SameAsDepth) {
        return;
      }

      if (node["source_camera_index"].isInt()) {
        const int source_index = static_cast<int>(node["source_camera_index"]);
        OKVIS_ASSERT_TRUE(Exception, source_index >= 0
                                      && static_cast<size_t>(source_index)
                                             < viParameters_.nCameraSystem.numCameras(),
                          "mapping " << name << " source camera index is out of range")
        const auto source = viParameters_.nCameraSystem.cameraGeometry(
            static_cast<size_t>(source_index));
        const auto source_T_SC = viParameters_.nCameraSystem.T_SC(
            static_cast<size_t>(source_index));
        const kinematics::Transformation T_target_source =
            mapping_input.targetT_SC->inverse() * (*source_T_SC);
        const double rotation_error = Eigen::AngleAxisd(
            T_target_source.q()).angle();
        OKVIS_ASSERT_TRUE(
            Exception,
            T_target_source.r().norm() <= 1.0e-6
                && std::abs(rotation_error) <= 1.0e-6,
            "mapping " << name << " source camera " << source_index
                        << " is not in the target optical frame; depth registration "
                           "must be performed before mapping input preprocessing")
        modality.sourceGeometry =
            (modality.geometry == MappingImagePreprocessor::Geometry::Pinhole)
                ? makeNoDistortionCamera(*source)
                : source;
        return;
      }

      const cv::FileNode source_node = node["source"];
      OKVIS_ASSERT_TRUE(Exception, source_node.isMap(),
                        "mapping " << name
                                    << " requires source_camera_index or source calibration")
      modality.sourceGeometry =
          makeExplicitMappingSourceCamera(source_node, modality.geometry);

      // the independent explicit calibration above remains authoritative.
      if (modality.stream.size() > 3
          && modality.stream.compare(0, 3, "cam") == 0) {
        const std::string suffix = modality.stream.substr(3);
        const bool numeric = std::all_of(
            suffix.begin(), suffix.end(),
            [](const unsigned char c) { return std::isdigit(c) != 0; });
        if (numeric) {
          const size_t source_index = static_cast<size_t>(std::stoul(suffix));
          OKVIS_ASSERT_TRUE(
              Exception,
              source_index < viParameters_.nCameraSystem.numCameras(),
              "mapping " << name << " stream " << modality.stream
                          << " refers to an out-of-range camera")
          const auto source_T_SC =
              viParameters_.nCameraSystem.T_SC(source_index);
          const kinematics::Transformation T_target_source =
              mapping_input.targetT_SC->inverse() * (*source_T_SC);
          const double rotation_error =
              Eigen::AngleAxisd(T_target_source.q()).angle();
          OKVIS_ASSERT_TRUE(
              Exception,
              T_target_source.r().norm() <= 1.0e-6
                  && std::abs(rotation_error) <= 1.0e-6,
              "mapping " << name << " stream " << modality.stream
                          << " is not in the target optical frame; depth "
                             "registration must be performed before mapping "
                             "input preprocessing")
        }
      }
    };

    parse_modality("rgb", mapping_input.rgb);
    parse_modality("depth", mapping_input.depth);
  }

  viParameters_.mappingInputPreprocessor =
      std::make_shared<MappingImagePreprocessor>(std::move(mapping_input));
  if (viParameters_.mappingInputPreprocessor->enabled()) {
    LOG(INFO) << "Mapping input preprocessing enabled: target camera "
              << viParameters_.mappingInputPreprocessor->targetCameraIndex()
              << ", RGB stream="
              << viParameters_.mappingInputPreprocessor->rgbStream()
              << ", depth stream="
              << viParameters_.mappingInputPreprocessor->depthStream();
  }

  if(file["lidar"].isMap()){
    viParameters_.lidar = okvis::LidarParameters(); 
    if(!getLiDARCalibration(file["lidar"], *viParameters_.lidar)){
      LOG(ERROR) << "Could not parse the LiDAR config file";
    } else {
      std::stringstream s;
      s << (*viParameters_.lidar).T_SL.T();
      LOG(INFO) << "Parsed LiDAR with the following characteristics: \n" 
                << "elevation_resolution_angle: " << std::to_string((*viParameters_.lidar).elevation_resolution_angle) << " \n"
                << "azimuth_resolution_angle: " << std::to_string((*viParameters_.lidar).azimuth_resolution_angle) << "\n"
                << "T_SL:\n " << s.str();
    }
  } else {
    LOG(INFO) << "No LiDAR declared";
  }

  parseEntry(file["camera_parameters"], "timestamp_tolerance",
             viParameters_.camera.timestamp_tolerance);
  cv::FileNode T = file["camera_parameters"]["sync_cameras"];
  OKVIS_ASSERT_TRUE(
    Exception, T.isSeq(),
    "missing real array parameter " << "camera_parameters" << ": " << "sync_cameras")
  for(auto iter = T.begin(); iter != T.end(); ++iter) {
    viParameters_.camera.sync_cameras.insert(int(*iter));
  }
  parseEntry(file["camera_parameters"], "image_delay",
             viParameters_.camera.image_delay);
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics",
             viParameters_.camera.online_calibration.do_extrinsics);
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics_final_ba",
             viParameters_.camera.online_calibration.do_extrinsics_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r",
             viParameters_.camera.online_calibration.sigma_r);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha",
             viParameters_.camera.online_calibration.sigma_alpha);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r_final_ba",
             viParameters_.camera.online_calibration.sigma_r_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha_final_ba",
             viParameters_.camera.online_calibration.sigma_alpha_final_ba);

  parseEntry(file["imu_parameters"], "use",
             viParameters_.imu.use);
  Eigen::Matrix4d T_BS;
  parseEntry(file["imu_parameters"], "T_BS", T_BS);
  viParameters_.imu.T_BS = kinematics::Transformation(T_BS);
  parseEntry(file["imu_parameters"], "a_max",
             viParameters_.imu.a_max);
  parseEntry(file["imu_parameters"], "g_max",
             viParameters_.imu.g_max);
  parseEntry(file["imu_parameters"], "sigma_g_c",
             viParameters_.imu.sigma_g_c);
  parseEntry(file["imu_parameters"], "sigma_bg",
             viParameters_.imu.sigma_bg);
  parseEntry(file["imu_parameters"], "sigma_a_c",
             viParameters_.imu.sigma_a_c);
  parseEntry(file["imu_parameters"], "sigma_ba",
             viParameters_.imu.sigma_ba);
  parseEntry(file["imu_parameters"], "sigma_gw_c",
             viParameters_.imu.sigma_gw_c);
  parseEntry(file["imu_parameters"], "sigma_aw_c",
             viParameters_.imu.sigma_aw_c);
  parseEntry(file["imu_parameters"], "a0",
             viParameters_.imu.a0);
  parseEntry(file["imu_parameters"], "g0",
             viParameters_.imu.g0);
  parseEntry(file["imu_parameters"], "g",
             viParameters_.imu.g);
  parseEntry(file["imu_parameters"], "s_a",
             viParameters_.imu.s_a);

  parseEntry(file["frontend_parameters"], "detection_threshold",
             viParameters_.frontend.detection_threshold);
  parseEntry(file["frontend_parameters"], "absolute_threshold",
             viParameters_.frontend.absolute_threshold);
  parseEntry(file["frontend_parameters"], "matching_threshold",
             viParameters_.frontend.matching_threshold);
  parseEntry(file["frontend_parameters"], "octaves",
             viParameters_.frontend.octaves);
  parseEntry(file["frontend_parameters"], "max_num_keypoints",
             viParameters_.frontend.max_num_keypoints);
  parseEntry(file["frontend_parameters"], "keyframe_overlap",
             viParameters_.frontend.keyframe_overlap);

  parseEntry(file["frontend_parameters"], "parallelise_detection",
             viParameters_.frontend.parallelise_detection);
  parseEntry(file["frontend_parameters"], "num_matching_threads",
             viParameters_.frontend.num_matching_threads);

  parseEntry(file["estimator_parameters"], "num_keyframes",
             viParameters_.estimator.num_keyframes);
  parseEntry(file["estimator_parameters"], "num_loop_closure_frames",
             viParameters_.estimator.num_loop_closure_frames);
  parseEntry(file["estimator_parameters"], "num_imu_frames",
             viParameters_.estimator.num_imu_frames);
  parseEntry(file["estimator_parameters"], "do_loop_closures",
             viParameters_.estimator.do_loop_closures);
  parseEntry(file["estimator_parameters"], "do_final_ba",
             viParameters_.estimator.do_final_ba);
  parseEntry(file["estimator_parameters"], "enforce_realtime",
             viParameters_.estimator.enforce_realtime);
  parseEntry(file["estimator_parameters"], "realtime_min_iterations",
             viParameters_.estimator.realtime_min_iterations);
  parseEntry(file["estimator_parameters"], "realtime_max_iterations",
             viParameters_.estimator.realtime_max_iterations);
  parseEntry(file["estimator_parameters"], "realtime_time_limit",
             viParameters_.estimator.realtime_time_limit);
  parseEntry(file["estimator_parameters"], "realtime_num_threads",
             viParameters_.estimator.realtime_num_threads);
  parseEntry(file["estimator_parameters"], "full_graph_iterations",
             viParameters_.estimator.full_graph_iterations);
  parseEntry(file["estimator_parameters"], "full_graph_num_threads",
             viParameters_.estimator.full_graph_num_threads);
  parseEntry(file["estimator_parameters"], "p_dbow",
             viParameters_.estimator.p_dbow);
  parseEntry(file["estimator_parameters"], "drift_percentage_heuristic",
             viParameters_.estimator.drift_percentage_heuristic);

  if (file["output_parameters"]["display_topview"].isInt()
      || file["output_parameters"]["display_topview"].isString()) {
    parseEntry(file["output_parameters"], "display_topview",
               viParameters_.output.display_topview);
  } else {
    viParameters_.output.display_topview = false;
  }
  parseEntry(file["output_parameters"], "display_matches",
             viParameters_.output.display_matches);
  parseEntry(file["output_parameters"], "display_overhead",
             viParameters_.output.display_overhead);
  parseEntry(file["output_parameters"], "enable_submapping",
             viParameters_.output.enable_submapping);

  if(file["gps_parameters"].isMap()){
    viParameters_.gps = okvis::GpsParameters(); 
    if(!getGpsCalibration(file["gps_parameters"], *viParameters_.gps)){
      LOG(ERROR) << "Could not parse the GPS config";
    } else {
      LOG(INFO) << "Parsed GPS with the following characteristics: \n" 
                << "\tdata_type: " << (*viParameters_.gps).type << " \n"
                << "\tr_SA: " << (*viParameters_.gps).r_SA.transpose() << " \n"
                << "\tyaw_error_threshold: " << std::to_string((*viParameters_.gps).yawErrorThreshold) << " \n"
                << "\trobust_gps_init: " << std::boolalpha << (*viParameters_.gps).robustGpsInit;
    }
  } else {
    LOG(INFO) << "No GPS declared";
  }

  readConfigFile_ = true;
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, int& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isInt(),
                    "missing integer parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

void ViParametersReader::parseEntry(const cv::FileNode &file,
                                    std::string name, double& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isReal(),
                    "missing real parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, bool& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isInt() || file[name].isString(),
                    "missing boolean parameter " << file.name() << ": " << name)
  if (file[name].isInt()) {
    readValue = int(file[name]) != 0;
    return;
  }
  if (file[name].isString()) {
    std::string str = std::string(file[name]);

    str = str.substr(0,str.find(" "));

    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    if (str.compare("false")  == 0
        || str.compare("no")  == 0
        || str.compare("n")   == 0
        || str.compare("off") == 0) {
      readValue = false;
      return;
    }
    if (str.compare("true")   == 0
        || str.compare("yes") == 0
        || str.compare("y")   == 0
        || str.compare("on")  == 0) {
      readValue = true;
      return;
    }
    OKVIS_THROW(Exception, "Boolean with uninterpretable value " << str)
  }
  return;
}

void ViParametersReader::parseEntry(const cv::FileNode &file, std::string name,
                                    Eigen::Matrix4d& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2], T[3],
               T[4], T[5], T[6], T[7],
               T[8], T[9], T[10], T[11],
               T[12], T[13], T[14], T[15];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    Eigen::Vector3d& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    std::string& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isString(),
                    "missing string parameter " << file.name() << ": " << name)
  readValue = std::string(T);
}

bool ViParametersReader::getCameraCalibration(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileStorage& configurationFile) {

  bool success = getCalibrationViaConfig(calibrations, configurationFile["cameras"]);
  return success;
}

bool ViParametersReader::getCalibrationViaConfig(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileNode cameraNode) const {

  calibrations.clear();
  bool gotCalibration = false;

  if (cameraNode.isSeq()
     && cameraNode.size() > 0) {
    size_t camIdx = 0;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      if ((*it).isMap()
          && (*it)["T_SC"].isSeq()
          && (*it)["image_dimension"].isSeq()
          && (*it)["image_dimension"].size() == 2
          && (*it)["distortion_coefficients"].isSeq()
          && (*it)["distortion_coefficients"].size() >= 4
          && (*it)["distortion_type"].isString()
          && (*it)["focal_length"].isSeq()
          && (*it)["focal_length"].size() == 2
          && (*it)["principal_point"].isSeq()
          && (*it)["principal_point"].size() == 2) {
        LOG(INFO) << "Found calibration in configuration file for camera " << camIdx;
        gotCalibration = true;
      } else {
        LOG(WARNING) << "Found incomplete calibration in configuration file for camera " << camIdx
                     << ". Will not use the calibration from the configuration file.";
        return false;
      }

      ++camIdx;
    }
  }
  else
    LOG(INFO) << "Did not find a calibration in the configuration file.";

  if (gotCalibration) {
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {

      CameraCalibration calib;

      if((std::string)((*it)["cam_model"]) == "pinhole"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode distortionCoefficientNode = (*it)["distortion_coefficients"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];

        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.distortionCoefficients.resize(int(distortionCoefficientNode.size()));
        for(int i=0; i<int(distortionCoefficientNode.size()); ++i) {
          calib.distortionCoefficients[i] = distortionCoefficientNode[i];
        }
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.distortionType = std::string((*it)["distortion_type"]);
        calib.cameraModel = "pinhole";
      } else if ( (std::string)((*it)["cam_model"]) == "eucm"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];
        cv::FileNode eucmParamNode= (*it)["eucm_parameters"];

        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = okvis::kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.eucmParameters << eucmParamNode[0], eucmParamNode[1];
        calib.distortionType = "None";
        calib.cameraModel = "eucm";
      }

      if((*it)["camera_type"].isString()){
        std::string camera_type = std::string((*it)["camera_type"]);
        if(camera_type.compare(0,4,"gray")==0) {
          calib.cameraType.isColour = false;
        } else {
          calib.cameraType.isColour = true;
        }
        if(camera_type.size()>=6){
          if(camera_type.compare(camera_type.size()-6,6,"+depth")==0) {
            calib.cameraType.depthType.isDepthCamera = true;
          } else {
            calib.cameraType.depthType.isDepthCamera = false;
          }
        }
      }
      if((*it)["slam_use"].isString()){
        std::string slam_use = std::string((*it)["slam_use"]);
        if(slam_use.compare(0,5,"okvis")==0) {
          calib.cameraType.isUsed = true;
        } else {
          calib.cameraType.isUsed = false;
        }
        if(slam_use.size()>=6){
          if(slam_use.compare(slam_use.size()-6,6,"-depth")==0) {
            calib.cameraType.depthType.createDepth = true;
          } else {
            calib.cameraType.depthType.createDepth = false;
          }
        }
        if(slam_use.size()>=8){
          if(slam_use.compare(slam_use.size()-8,8,"-virtual")==0) {
            calib.cameraType.depthType.createVirtual = true;
          } else {
            calib.cameraType.depthType.createVirtual = false;
          }
        }
      }
      if((*it)["sigma_pixels"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_pixels"]);
      }
      if((*it)["sigma_depth"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_depth"]);
      }

      if((*it)["mapping"].isString() && (std::string)((*it)["mapping"]) == "true"){
        calib.cameraType.isUsedMapping = true;
      }


      calibrations.push_back(calib);
    }
  }
  return gotCalibration;
}

bool ViParametersReader::getLiDARCalibration(const cv::FileNode& calibrationNode, okvis::LidarParameters& lidarParameters){

  if(!calibrationNode["T_SL"].isSeq() || !calibrationNode["elevation_resolution_angle"].isReal() || !calibrationNode["azimuth_resolution_angle"].isReal()){
    return false;
  }

  cv::FileNode T_SL_node = calibrationNode["T_SL"];
  Eigen::Matrix4d T_SL;
  T_SL << T_SL_node[0], T_SL_node[1], T_SL_node[2], T_SL_node[3],
          T_SL_node[4], T_SL_node[5], T_SL_node[6], T_SL_node[7],
          T_SL_node[8], T_SL_node[9], T_SL_node[10], T_SL_node[11],
          T_SL_node[12], T_SL_node[13], T_SL_node[14], T_SL_node[15];
  lidarParameters.T_SL = okvis::kinematics::Transformation(T_SL);
  lidarParameters.elevation_resolution_angle = calibrationNode["elevation_resolution_angle"];
  lidarParameters.azimuth_resolution_angle = calibrationNode["azimuth_resolution_angle"];

  return true;
}

bool ViParametersReader::getGpsCalibration(const cv::FileNode& calibrationNode, okvis::GpsParameters& gpsParameters){

  parseEntry(calibrationNode, "data_type",
             gpsParameters.type);
  parseEntry(calibrationNode, "r_SA",
             gpsParameters.r_SA);
  parseEntry(calibrationNode, "yaw_error_threshold",
             gpsParameters.yawErrorThreshold);

  if (calibrationNode["robust_gps_init"].isInt()
      || calibrationNode["robust_gps_init"].isString()) {
    parseEntry(calibrationNode, "robust_gps_init",
               gpsParameters.robustGpsInit);
  } else {
    gpsParameters.robustGpsInit = false;
  }

  return true;
}


}  
