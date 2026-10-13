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
 * @file FrameRelativeAdapter.hpp
 * @brief Header file for the FrameRelativeAdapter class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMERELATIVEADAPTER_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMERELATIVEADAPTER_HPP_

#include <stdlib.h>
#include <vector>
#include <opengv/types.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/assert_macros.hpp>

namespace opengv {

namespace relative_pose {

class FrameRelativeAdapter : public RelativeAdapterBase {
 private:
  using RelativeAdapterBase::_t12;
  using RelativeAdapterBase::_R12;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  using Estimator = okvis::ViSlamBackend;

  FrameRelativeAdapter(const Estimator & estimator,
                       const okvis::cameras::NCameraSystem & nCameraSystem,
                       uint64_t multiFrameIdA, size_t camIdA,
                       uint64_t multiFrameIdB, size_t camIdB);

  virtual ~FrameRelativeAdapter() {
  }

  virtual opengv::bearingVector_t getBearingVector1(size_t index) const;

  virtual opengv::bearingVector_t getBearingVector2(size_t index) const;

  virtual opengv::translation_t getCamOffset1(size_t index) const;

  virtual opengv::rotation_t getCamRotation1(size_t index) const;

  virtual opengv::translation_t getCamOffset2(size_t index) const;

  virtual opengv::rotation_t getCamRotation2(size_t index) const;

  virtual size_t getNumberCorrespondences() const;

  double getSigmaAngle1(size_t index);

  double getSigmaAngle2(size_t index);

  size_t getMatchKeypointIdxA(size_t index) {
    return matches_.at(index).idxA;
  }

  size_t getMatchKeypointIdxB(size_t index) {
    return matches_.at(index).idxB;
  }

  virtual double getWeight(size_t) const {
    return 1.0;
  }  

 private:

  opengv::bearingVectors_t bearingVectors1_;

  opengv::bearingVectors_t bearingVectors2_;

  okvis::Matches matches_;

  std::vector<double> sigmaAngles1_;

  std::vector<double> sigmaAngles2_;

};

}
}

#endif 
