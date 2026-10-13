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
 * @file FrameNoncentralAbsoluteAdapter.hpp
 * @brief Header file for the FrameNoncentralAbsoluteAdapter class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMENONCENTRALABSOLUTEADAPTER_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMENONCENTRALABSOLUTEADAPTER_HPP_

#include <stdlib.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <opengv/types.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#pragma GCC diagnostic pop
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

namespace opengv {

namespace absolute_pose {

class FrameNoncentralAbsoluteAdapter : public AbsoluteAdapterBase {
 private:
  using AbsoluteAdapterBase::_t;
  using AbsoluteAdapterBase::_R;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  using Estimator = okvis::ViSlamBackend;

  typedef std::vector<int> matches_t;

  FrameNoncentralAbsoluteAdapter(
      const Estimator & estimator,
      const okvis::cameras::NCameraSystem & nCameraSystem,
      std::shared_ptr<okvis::MultiFrame> frame);

  virtual ~FrameNoncentralAbsoluteAdapter() override = default;

  virtual opengv::bearingVector_t getBearingVector(size_t index) const override final;

  virtual opengv::translation_t getCamOffset(size_t index) const override final;

  virtual opengv::rotation_t getCamRotation(size_t index) const override final;

  virtual opengv::point_t getPoint(size_t index) const override final;

  virtual size_t getNumberCorrespondences() const override final;

  int camIndex(size_t index) const {return int(camIndices_.at(index));}

  int keypointIndex(size_t index) const {return int(keypointIndices_.at(index));}

  virtual double getWeight(size_t) const  override final {
    return 1.0;
  }  

  double getSigmaAngle(size_t index);

 private:

  opengv::bearingVectors_t bearingVectors_;

  opengv::points_t points_;

  std::vector<size_t> camIndices_;

  std::vector<size_t> keypointIndices_;

  opengv::translations_t camOffsets_;

  opengv::rotations_t camRotations_;

  std::vector<double> sigmaAngles_;

};

}
}

#endif 
