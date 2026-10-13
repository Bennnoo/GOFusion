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
 * @file ViSensorBase.hpp
 * @brief Header file for the ViSensorBase class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_VISENSORBASE_HPP_
#define INCLUDE_OKVIS_VISENSORBASE_HPP_

#include <Eigen/Core>
#include <map>

#include <opencv2/core.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace okvis {

class ViSensorBase {
public:

  typedef std::function<
        bool(const okvis::Time &,
             const std::map<size_t, cv::Mat> &,
             const std::map<size_t, cv::Mat> &)> ImageCallback;

  typedef std::function<
        bool(const okvis::Time &,
             const Eigen::Vector3d &, const Eigen::Vector3d & )> ImuCallback;
  typedef std::function<
        bool(const okvis::Time &,
             const Eigen::Vector3d &, const Eigen::Vector3d &)> GpsCallback;
  typedef std::function<
          bool(const okvis::Time &,
               const double &, const double &, const double &,
               const double &, const double &)> GeodeticGpsCallback;

  typedef std::function<
          bool(std::map<size_t, std::vector<okvis::CameraMeasurement>>&)> DepthImageCallback;

  virtual ~ViSensorBase() = default;

  virtual void setImagesCallback(const ImageCallback& imagesCallback) final {
    imagesCallbacks_.push_back(imagesCallback);
  }

  virtual void setImuCallback(const ImuCallback& imuCallback) final {
    imuCallbacks_.push_back(imuCallback);
  }

  virtual void setGpsCallback(const GpsCallback& gpsCallback) final {
    gpsCallback_ = gpsCallback;
  }

  virtual void setGeodeticGpsCallback(const GeodeticGpsCallback & geodeticGpsCallback) final {
    geodeticGpsCallback_ = geodeticGpsCallback;
  }

  virtual void setDepthImageCallback(const DepthImageCallback & depthImageCallback) final {
    depthCallback_ = depthImageCallback;
  }

  virtual bool startStreaming() = 0;

  virtual bool stopStreaming() = 0;

  virtual bool isStreaming() = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
  AlignedVector<ImageCallback> imagesCallbacks_; 
  AlignedVector<ImuCallback> imuCallbacks_; 
  GpsCallback gpsCallback_; 
  GeodeticGpsCallback geodeticGpsCallback_; 
  DepthImageCallback depthCallback_;  
};

class DatasetReaderBase : public ViSensorBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  const uint64 GNSS_LEAP_NANOSECONDS = 0;

  virtual bool setDatasetPath(const std::string & path) = 0;

  virtual bool setStartingDelay(const okvis::Duration & deltaT) = 0;

  virtual double completion() const = 0;

};

} 

#endif
