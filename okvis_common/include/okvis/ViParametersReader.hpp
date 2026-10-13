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
 */

/**
 * @file ViParametersReader.hpp
 * @brief Header file for the ViParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIPARAMETERSREADER_HPP_
#define INCLUDE_OKVIS_VIPARAMETERSREADER_HPP_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop
#include <opencv2/calib3d.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

namespace okvis {

class ViParametersReader{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  ViParametersReader();

  ViParametersReader(const std::string& filename);

  ~ViParametersReader() = default;

  void readConfigFile(const std::string& filename);

  bool getParameters(okvis::ViParameters& parameters) const{
    if(readConfigFile_)
      parameters = viParameters_;
    return readConfigFile_;
  }

 protected:

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

  static void parseEntry(const cv::FileNode &file, std::string name, int& readValue);

  static void parseEntry(const cv::FileNode& file, std::string name, double& readValue);

  static void parseEntry(const cv::FileNode &file, std::string name, bool& readValue);

  static void parseEntry(const cv::FileNode& file, std::string name, Eigen::Matrix4d& readValue);

  static void parseEntry(const cv::FileNode &file, std::string name, Eigen::Vector3d& readValue);

  static void parseEntry(const cv::FileNode &file, std::string name, std::string& readValue);

  bool readConfigFile_; 
  okvis::ViParameters viParameters_;   

  bool getCameraCalibration(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileStorage& configurationFile);


  bool getCalibrationViaConfig(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileNode cameraNode) const;

  bool getLiDARCalibration(const cv::FileNode& calibrationNode, okvis::LidarParameters& lidarParameters);

  bool getGpsCalibration(const cv::FileNode& calibrationNode, okvis::GpsParameters& gpsParameters);

};

}

#endif 
