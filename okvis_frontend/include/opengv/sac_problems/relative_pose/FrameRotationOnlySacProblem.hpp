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
 * @file FrameRotationOnlySacProblem.hpp
 * @brief Header file for the FrameRotationOnlySacProblem class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMEROTATIONONLYSACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMEROTATIONONLYSACPROBLEM_HPP_

#include <okvis/assert_macros.hpp>
#include <opengv/types.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/relative_pose/RotationOnlySacProblem.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

namespace opengv {

namespace sac_problems {

namespace relative_pose {

class FrameRotationOnlySacProblem : public RotationOnlySacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef RotationOnlySacProblem base_t;

  using base_t::adapter_t;

  using base_t::model_t;

  FrameRotationOnlySacProblem(adapter_t & adapter)
      : base_t(adapter),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported")
  }

  FrameRotationOnlySacProblem(adapter_t & adapter,
                              const std::vector<int> & indices)
      : base_t(adapter, indices),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported")
  }

  virtual ~FrameRotationOnlySacProblem() override;

  virtual void getSelectedDistancesToModel(const model_t & model,
                                           const std::vector<int> & indices,
                                           std::vector<double> & scores) const override final {
    for (size_t i = 0; i < indices.size(); i++) {
      bearingVector_t f1 = adapterDerived_.getBearingVector1(size_t(indices[i]));
      bearingVector_t f2 = adapterDerived_.getBearingVector2(size_t(indices[i]));

      bearingVector_t f2_unrotated = model * f2;

      bearingVector_t f1_unrotated = model.transpose() * f1;

      point_t error1 = (f2_unrotated - f1);
      point_t error2 = (f1_unrotated - f2);
      double error_squared1 = error1.transpose() * error1;
      double error_squared2 = error2.transpose() * error2;
      scores.push_back(
          error_squared1 * 0.5 / adapterDerived_.getSigmaAngle1(size_t(indices[i]))
              + error_squared2 * 0.5
                  / adapterDerived_.getSigmaAngle2(size_t(indices[i])));
    }
  }

 protected:

  opengv::relative_pose::FrameRelativeAdapter & adapterDerived_;

};

FrameRotationOnlySacProblem::~FrameRotationOnlySacProblem() {}

}
}
}

#endif 
