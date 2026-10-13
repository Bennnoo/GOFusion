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
 * @file ReprojectionErrorBase.hpp
 * @brief Header file for the ReprojectionErrorBase class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_REPROJECTIONERRORBASE_HPP_
#define INCLUDE_OKVIS_CERES_REPROJECTIONERRORBASE_HPP_

#include <ceres/sized_cost_function.h>

#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {

namespace ceres {

template<int ...T>
class ReprojectionErrorBase_ :
    public ::ceres::SizedCostFunction<T...>,
    public ErrorInterface {
 public:

  typedef ::ceres::SizedCostFunction<T...> ceres_base_t;

  uint64_t cameraId() const {
    return cameraId_;
  }

  void setCameraId(uint64_t cameraId) {
    cameraId_ = cameraId;
  }

  uint64_t cameraId_; 
};

template<int ...T>
class ReprojectionError2dBase_ : public ReprojectionErrorBase_<T...> {
 public:

  typedef ReprojectionErrorBase_<T...> base_t;

  typedef typename ReprojectionErrorBase_<T...>::ceres_base_t ceres_base_t;

  typedef Eigen::Vector2d measurement_t;

  typedef Eigen::Matrix2d covariance_t;

  virtual void setMeasurement(const measurement_t& measurement) = 0;

  virtual void setInformation(const covariance_t& information) = 0;

  virtual const measurement_t& measurement() const = 0;

  virtual const covariance_t& information() const = 0;

  virtual const covariance_t& covariance() const = 0;

  virtual std::shared_ptr<ReprojectionError2dBase_<T...>> clone() const = 0;

};

typedef ReprojectionErrorBase_<
2 ,
7 ,
4 ,
7 >
ReprojectionErrorBase;

typedef ReprojectionError2dBase_<
2 ,
7 ,
4 ,
7 >
ReprojectionError2dBase;

}

}

#endif 
