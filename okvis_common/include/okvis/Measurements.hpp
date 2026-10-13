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
 * @file Measurements.hpp
 * @brief This file contains the templated measurement structs, structs encapsulating
 *        Sensor data and related typedefs.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_MEASUREMENTS_HPP_
#define INCLUDE_OKVIS_MEASUREMENTS_HPP_

#include <cstddef>
#include <deque>
#include <vector>
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <map>
#include <Eigen/Dense>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

enum class FrameBufferMemoryKind {
  OrdinaryHost,
  CudaRegisteredMapped,
  CudaPinnedStaging,
};

class FrameBufferLifetime {
public:
  virtual ~FrameBufferLifetime() = default;
  virtual void recordCudaUse(void* native_stream) noexcept = 0;
  virtual FrameBufferMemoryKind memoryKind() const noexcept = 0;
  virtual const void* hostData() const noexcept = 0;
  virtual const void* mappedDeviceData() const noexcept = 0;
  virtual size_t capacityBytes() const noexcept = 0;
};

template<class MEASUREMENT_T>
struct Measurement {
  okvis::Time timeStamp;      
  MEASUREMENT_T measurement;  
  int sensorId = -1;          

  Measurement(const Measurement<MEASUREMENT_T>& other) noexcept :
    timeStamp(other.timeStamp), measurement(other.measurement), sensorId(other.sensorId)
  {

  }

  Measurement<MEASUREMENT_T>& operator=(const Measurement<MEASUREMENT_T>& other) noexcept {
    timeStamp = other.timeStamp;
    measurement = other.measurement;
    sensorId = other.sensorId;
    return *this;
  }

  Measurement()
      : timeStamp(0.0) {
  }

  Measurement(const okvis::Time& timeStamp_, const MEASUREMENT_T& measurement_,
              int sensorId = -1)
      : timeStamp(timeStamp_),
        measurement(measurement_),
        sensorId(sensorId) {
  }
};

struct ImuSensorReadings {

  ImuSensorReadings()
      : gyroscopes(),
        accelerometers() {
  }

  ImuSensorReadings(const Eigen::Matrix<double,3,1,Eigen::DontAlign>& gyroscopes_,
                    const Eigen::Matrix<double,3,1,Eigen::DontAlign>& accelerometers_)
      : gyroscopes(gyroscopes_),
        accelerometers(accelerometers_) {
  }
  Eigen::Matrix<double,3,1,Eigen::DontAlign> gyroscopes;     
  Eigen::Matrix<double,3,1,Eigen::DontAlign> accelerometers; 
};

struct LidarSensorReadings {

    LidarSensorReadings()
            : rayMeasurement(),
              intensity() {
    }

    LidarSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& rayMeasurement_,
                        unsigned int& intensity_)
            : rayMeasurement(rayMeasurement_),
              intensity(intensity_) {
    }
    LidarSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& rayMeasurement_)
            : rayMeasurement(rayMeasurement_),
              intensity(0) {
    }

    Eigen::Matrix<double,3,1,Eigen::Unaligned> rayMeasurement;  
    unsigned int intensity; 
};

struct CameraData {


  cv::Mat image;  
  cv::Mat depthImage;  

  std::shared_ptr<FrameBufferLifetime> imageBufferLifetime;
  std::shared_ptr<FrameBufferLifetime> depthBufferLifetime;
  std::vector<cv::KeyPoint> keypoints; 

  uint64_t imageGeometryRevision = 0;
  uint64_t depthGeometryRevision = 0;
  bool deliversKeypoints; 
};

typedef Measurement<ImuSensorReadings> ImuMeasurement;

typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurementDeque;

typedef Measurement<LidarSensorReadings> LidarMeasurement;

typedef std::deque<LidarMeasurement, Eigen::aligned_allocator<LidarMeasurement> > LidarMeasurementDeque;

typedef Measurement<CameraData> CameraMeasurement;

typedef std::deque<CameraMeasurement, Eigen::aligned_allocator<CameraMeasurement> > DepthMeasurementDeque;

struct KeypointData {
  std::vector<cv::KeyPoint> keypoints;  
  std::vector<long unsigned int> landmarkIds; 
  cv::Mat descriptors;  
};

struct FrameData {

  typedef std::shared_ptr<okvis::FrameData> Ptr;
  CameraData image; 
  KeypointData keypoints; 
};

struct GpsSensorReadings {

  GpsSensorReadings()
      : position(),
        covariances() {
  }

  GpsSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& position_,
                    const Eigen::Matrix<double,3,3,Eigen::Unaligned>& covariances_)
      : position(position_),
        covariances(covariances_){

    latitude = -1.;
    longitdue = -1.;
    height = -1.;
    horizontalAccuracy = -1.;
    verticalAccuracy = -1.;
  }

  GpsSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& position_,
                    const double sigma_x,
                    const double sigma_y,
                    const double sigma_z){
    this->position = position_;
    Eigen::Matrix<double,3,3,Eigen::Unaligned> covMat;
    covMat.setIdentity();
    covMat(0,0) = sigma_x * sigma_x;
    covMat(1,1) = sigma_y * sigma_y;
    covMat(2,2) = sigma_z * sigma_z;
    this->covariances = covMat;
  }

  GpsSensorReadings(double lat, double lon, double height,
                    double hAcc, double vAcc)
          : latitude(lat), longitdue(lon), height(height),
            horizontalAccuracy(hAcc), verticalAccuracy(vAcc){

    position = Eigen::Matrix<double,3,1,Eigen::Unaligned>(0., 0., 0.);
    Eigen::Matrix<double,3,3,Eigen::Unaligned> covMat;
    covMat.setIdentity();
    covMat(0,0) = horizontalAccuracy * horizontalAccuracy;
    covMat(1,1) = horizontalAccuracy * horizontalAccuracy;
    covMat(2,2) = verticalAccuracy * verticalAccuracy;
    this->covariances = covMat;
  }

  void setPosition(double x, double y, double z){
    position.x() = x;
    position.y() = y;
    position.z() = z;
  }

  Eigen::Matrix<double,3,1,Eigen::Unaligned> position;     
  Eigen::Matrix<double,3,3,Eigen::Unaligned> covariances;     

  double latitude = 0.0; 
  double longitdue = 0.0; 
  double height = 0.0; 
  double horizontalAccuracy = 0.0; 
  double verticalAccuracy = 0.0; 

};
typedef Measurement<GpsSensorReadings> GpsMeasurement;
typedef std::deque<GpsMeasurement, Eigen::aligned_allocator<GpsMeasurement> > GpsMeasurementDeque;

}  

#endif 
