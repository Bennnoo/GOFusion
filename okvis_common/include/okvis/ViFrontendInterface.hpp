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
 * @file ViFrontendInterface.hpp
 * @brief Header file for the VioFrontendInterface class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIOFRONTENDINTERFACE_HPP_
#define INCLUDE_OKVIS_VIOFRONTENDINTERFACE_HPP_

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <okvis/kinematics/Transformation.hpp>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>

namespace okvis {

class ViSlamBackend;

using Estimator = ViSlamBackend;

class ViFrontendInterface {
 public:

  ViFrontendInterface() = default;

  virtual ~ViFrontendInterface() = default;

  virtual bool detectAndDescribe(
      size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut,
      const okvis::kinematics::Transformation& T_WC,
      const std::vector<cv::KeyPoint> * keypoints) = 0;

  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const okvis::ViParameters & params,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) = 0;

  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const = 0;

};

}

#endif 
