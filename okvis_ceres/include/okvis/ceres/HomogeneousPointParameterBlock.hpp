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
 * @file HomogeneousPointParameterBlock.hpp
 * @brief Header file for the HomogeneousPointParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTPARAMETERBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTPARAMETERBLOCK_HPP_

#include <okvis/ceres/ParameterBlockSized.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <Eigen/Core>

namespace okvis {

namespace ceres {

class HomogeneousPointParameterBlock : public ParameterBlockSized<4, 3,
    Eigen::Vector4d>
{
 public:

  typedef Eigen::Vector4d estimate_t;

  typedef ParameterBlockSized<4, 3, estimate_t> base_t;

  HomogeneousPointParameterBlock();

  HomogeneousPointParameterBlock(const Eigen::Vector4d& point, uint64_t id,
                                 bool initialized = true);

  HomogeneousPointParameterBlock(const Eigen::Vector3d& point, uint64_t id,
                                 bool initialized = true);

  virtual ~HomogeneousPointParameterBlock() override;

  void setInitialized(bool initialized)
  {
    initialized_ = initialized;
  }

  virtual void setParameters(const double* parameters) override {
    estimate_ = Eigen::Map<const Eigen::Matrix<double,4,1>>(parameters);
  }

  virtual double* parameters() override {
    return estimate_.data();
  }

  virtual const double* parameters() const override {
    return estimate_.data();
  }

  bool initialized() const
  {
    return initialized_;
  }

  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const override
  {
    HomogeneousPointManifold::plus(x0, Delta_Chi, x0_plus_Delta);
  }

  virtual void plusJacobian(const double* x0, double* jacobian) const override
  {
    HomogeneousPointManifold::plusJacobian(x0, jacobian);
  }

  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const override
  {
    HomogeneousPointManifold::minus(x0_plus_Delta, x0, Delta_Chi);
  }

  virtual void liftJacobian(const double* x0, double* jacobian) const override
  {
    HomogeneousPointManifold::minusJacobian(x0, jacobian);
  }

  virtual std::string typeInfo() const override
  {
    return "HomogeneousPointParameterBlock";
  }

 private:
  bool initialized_;  

};

}  
}  

#endif 
