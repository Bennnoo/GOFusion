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
 * @file SpeedAndBiasParameterBlock.hpp
 * @brief Header file for the SpeedAndBiasParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_SPEEDANDBIASPARAMETERBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_SPEEDANDBIASPARAMETERBLOCK_HPP_

#include <okvis/ceres/ParameterBlockSized.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <Eigen/Core>
#include <okvis/Time.hpp>

namespace okvis {

namespace ceres {

typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;

class SpeedAndBiasParameterBlock :
    public ParameterBlockSized<9, 9, SpeedAndBias> {
 public:

  typedef ParameterBlockSized<9, 9, SpeedAndBias> base_t;

  typedef SpeedAndBias estimate_t;

  SpeedAndBiasParameterBlock();

  SpeedAndBiasParameterBlock(const SpeedAndBias& speedAndBias, uint64_t id,
                             const okvis::Time& timestamp = okvis::Time(0));

  virtual ~SpeedAndBiasParameterBlock() override = default;

  void setTimestamp(const okvis::Time& timestamp) {
    timestamp_ = timestamp;
  }

  virtual void setParameters(const double* parameters) override final {
    estimate_ = Eigen::Map<const SpeedAndBias>(parameters);
  }

  virtual double* parameters() override final {
    return estimate_.data();
  }

  virtual const double* parameters() const override final {
    return estimate_.data();
  }

  okvis::Time timestamp() const {
    return timestamp_;
  }

  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const override final {
    Eigen::Map<const Eigen::Matrix<double, 9, 1> > x0_(x0);
    Eigen::Map<const Eigen::Matrix<double, 9, 1> > Delta_Chi_(Delta_Chi);
    Eigen::Map<Eigen::Matrix<double, 9, 1> > x0_plus_Delta_(x0_plus_Delta);
    x0_plus_Delta_ = x0_ + Delta_Chi_;
  }

  virtual void plusJacobian(const double* ,
                            double* jacobian) const override final {
    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > identity(
        jacobian);
    identity.setIdentity();
  }

  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const override final {
    Eigen::Map<const Eigen::Matrix<double, 9, 1> > x0_(x0);
    Eigen::Map<Eigen::Matrix<double, 9, 1> > Delta_Chi_(Delta_Chi);
    Eigen::Map<const Eigen::Matrix<double, 9, 1> > x0_plus_Delta_(
        x0_plus_Delta);
    Delta_Chi_ = x0_plus_Delta_ - x0_;
  }

  virtual void liftJacobian(const double* ,
                            double* jacobian) const override final {
    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > identity(
        jacobian);
    identity.setIdentity();
  }

  virtual std::string typeInfo() const override final {
    return "SpeedAndBiasParameterBlock";
  }

 private:
  okvis::Time timestamp_; 
};

}  
}  

#endif 
