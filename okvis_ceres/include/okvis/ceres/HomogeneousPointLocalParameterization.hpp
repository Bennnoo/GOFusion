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
 * @file HomogeneousPointLocalParameterization.hpp
 * @brief Header file for the HomogeneousPointLocalParameterization class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTLOCALPARAMETERIZATION_HPP_
#define INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTLOCALPARAMETERIZATION_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/manifold.h>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>

namespace okvis {

namespace ceres {

class HomogeneousPointManifold : public ::ceres::Manifold {
 public:

  virtual ~HomogeneousPointManifold() override = default;

  virtual int AmbientSize() const override {return 4;}

  virtual int TangentSize() const override {return 3;}

  static bool plus(const double* x,
                   const double* delta,
                   double* x_plus_delta);

  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const override;

  static bool plusJacobian(const double* x, double* jacobian);

  virtual bool PlusJacobian(const double* x, double* jacobian) const override;

  static bool minus(const double* y,
                    const double* x,
                    double* y_minus_x);

  virtual bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override;

  static bool minusJacobian(const double* x, double* jacobian);

  virtual bool MinusJacobian(const double* x, double* jacobian) const override;
};

}  
}  

#endif 
