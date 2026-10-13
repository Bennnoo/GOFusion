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
 * @file FrameAbsolutePoseSacProblem.hpp
 * @brief Header file for the FrameAbsolutePoseSacProblem class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMEABSOLUTEPOSESACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMEABSOLUTEPOSESACPROBLEM_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <opengv/types.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#pragma GCC diagnostic pop
#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/LoopclosureNoncentralAbsoluteAdapter.hpp>
#include <okvis/assert_macros.hpp>

namespace opengv {

namespace sac_problems {

namespace absolute_pose {

template<class DERIVED_ADAPTER_T>
class FrameAbsolutePoseSacProblem : public AbsolutePoseSacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef AbsolutePoseSacProblem base_t;

  using base_t::adapter_t;

  using base_t::algorithm_t;

  using base_t::model_t;

  FrameAbsolutePoseSacProblem(adapter_t & adapter, algorithm_t algorithm)
      : base_t(adapter, algorithm),
        adapterDerived_(
            *static_cast<DERIVED_ADAPTER_T*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(&_adapter)
        || dynamic_cast<opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter*>(&_adapter),
        "unsupported adapter type")
  }

  FrameAbsolutePoseSacProblem(adapter_t & adapter, algorithm_t algorithm,
                              const std::vector<int> & indices)
      : base_t(adapter, algorithm, indices),
        adapterDerived_(
            *static_cast<DERIVED_ADAPTER_T*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(&_adapter)
        || dynamic_cast<opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter*>(&_adapter),
        "unsupported adapter type")
  }

  virtual ~FrameAbsolutePoseSacProblem() {
  }

  virtual void getSelectedDistancesToModel(const model_t & model,
                                           const std::vector<int> & indices,
                                           std::vector<double> & scores) const {

    model_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = model.block<3, 3>(0, 0).transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0) * model.col(3);

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;
    scores.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {

      p_hom.block<3, 1>(0, 0) = adapterDerived_.getPoint(indices[i]);

      const point_t bodyReprojection = inverseSolution * p_hom;
      point_t reprojection = adapterDerived_.getCamRotation(indices[i])
          .transpose()
          * (bodyReprojection - adapterDerived_.getCamOffset(indices[i]));
      reprojection = reprojection / reprojection.norm();

      const point_t error = (reprojection
          - adapterDerived_.getBearingVector(indices[i]));
      const double error_squared = error.transpose() * error;
      scores.push_back(error_squared / adapterDerived_.getSigmaAngle(indices[i]));
    }
  }

 protected:

  DERIVED_ADAPTER_T & adapterDerived_;

};

}
}
}

#endif 
