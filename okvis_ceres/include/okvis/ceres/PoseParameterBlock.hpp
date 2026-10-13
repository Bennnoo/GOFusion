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
 * @file PoseParameterBlock.hpp
 * @brief Header file for the PoseParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_POSEPARAMETERBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_POSEPARAMETERBLOCK_HPP_

#include <Eigen/Core>
#include <okvis/ceres/ParameterBlockSized.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>

namespace okvis {
namespace ceres{

class PoseParameterBlock :
    public ParameterBlockSized<7,6,okvis::kinematics::TransformationCacheless>{
public:

  typedef okvis::kinematics::TransformationCacheless estimate_t;

  typedef ParameterBlockSized<7,6,estimate_t> base_t;

  PoseParameterBlock();

  PoseParameterBlock(const okvis::kinematics::Transformation& T_WS, uint64_t id,
                     const okvis::Time& timestamp = okvis::Time(0));

  virtual ~PoseParameterBlock() override;

  void setTimestamp(const okvis::Time& timestamp){timestamp_=timestamp;}

  virtual void setParameters(const double* parameters) override final {
    estimate_.setCoeffs(Eigen::Map<const Eigen::Matrix<double,7,1>>(parameters));
  }

  virtual double* parameters() override final {
    return estimate_.coeffs().data();
  }

  virtual const double* parameters() const override final {
    return estimate_.coeffs().data();
  }

  okvis::Time timestamp() const {return timestamp_;}

  virtual void plus(
      const double* x0, const double* Delta_Chi, double* x0_plus_Delta) const override final {
    PoseManifold::plus(x0,Delta_Chi,x0_plus_Delta);
  }

  virtual void plusJacobian(const double* x0, double* jacobian) const override final {
    PoseManifold::plusJacobian(x0,jacobian);
  }

  virtual void minus(
      const double* x0, const double* x0_plus_Delta, double* Delta_Chi) const override final {
    PoseManifold::minus(x0_plus_Delta, x0, Delta_Chi);
  }

  virtual void liftJacobian(const double* x0, double* jacobian) const override final {
    PoseManifold::minusJacobian(x0,jacobian);
  }

  virtual std::string typeInfo() const override final {return "PoseParameterBlock";}

private:
  okvis::Time timestamp_; 
};

} 
} 

#endif 
