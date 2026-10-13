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
 * @file kinematics/Transformation.hpp
 * @brief Header file for the TransformationT class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_TRANSFORMATION_HPP_
#define INCLUDE_OKVIS_TRANSFORMATION_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace okvis {

namespace kinematics {

double sinc(double x);

Eigen::Quaterniond deltaQ(const Eigen::Vector3d& dAlpha);

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d & PhiVec);

template<bool CACHE_C = true>
class TransformationT
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  friend class TransformationT<!CACHE_C>;

  TransformationT();

  TransformationT(const TransformationT<CACHE_C> & other);

  TransformationT(const TransformationT<!CACHE_C> & other);

  TransformationT(TransformationT<CACHE_C> && other);

  TransformationT(TransformationT<!CACHE_C> && other);

  TransformationT(const Eigen::Vector3d & r_AB, const Eigen::Quaterniond& q_AB);

  explicit TransformationT(const Eigen::Matrix4d & T_AB);

  ~TransformationT();

  template<typename Derived_coeffs>
  bool setCoeffs(const Eigen::MatrixBase<Derived_coeffs> & coeffs);

  template<typename Derived_coeffs>
  bool setParameters(const Eigen::MatrixBase<Derived_coeffs> & parameters)
  {
    return setCoeffs(parameters);
  }

  Eigen::Matrix4d T() const;

  Eigen::Matrix3d C() const;

  const Eigen::Map<Eigen::Vector3d> & r() const;

  const Eigen::Map<Eigen::Quaterniond> & q() const;

  Eigen::Matrix<double, 3, 4> T3x4() const;

  const Eigen::Matrix<double, 7, 1> & coeffs() const
  {
    return parameters_;
  }

  Eigen::Matrix<double, 7, 1> & coeffs()
  {
    if(CACHE_C) throw(std::runtime_error("can't access writable parameters in cached mode."));
    return parameters_;
  }

  const Eigen::Matrix<double, 7, 1> & parameters() const
  {
    return parameters_;
  }

  Eigen::Matrix<double, 7, 1> & parameters()
  {
    if(CACHE_C) throw(std::runtime_error("can't access writable parameters in cached mode."));
    return parameters_;
  }

  const double* parameterPtr() const
  {
    return &parameters_[0];
  }

  void setRandom();

  void setRandom(double translationMaxMeters, double rotationMaxRadians);

  void set(const Eigen::Matrix4d & T_AB);

  void set(const Eigen::Vector3d & r_AB, const Eigen::Quaternion<double>& q_AB);

  void setIdentity();

  static TransformationT<CACHE_C> Identity();

  TransformationT<CACHE_C> inverse() const;

  TransformationT<CACHE_C> operator*(const TransformationT<CACHE_C> & rhs) const;

  Eigen::Vector3d operator*(const Eigen::Vector3d & rhs) const;

  Eigen::Vector4d operator*(const Eigen::Vector4d & rhs) const;

  TransformationT<CACHE_C>& operator=(const TransformationT<CACHE_C> & rhs);

  TransformationT<CACHE_C>& operator=(const TransformationT<!CACHE_C> & rhs);

  template<typename Derived_delta>
  bool oplus(const Eigen::MatrixBase<Derived_delta> & delta);

  template<typename Derived_delta, typename Derived_jacobian>
  bool oplus(const Eigen::MatrixBase<Derived_delta> & delta,
             const Eigen::MatrixBase<Derived_jacobian> & jacobian);

  template<typename Derived_jacobian>
  bool oplusJacobian(
      const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

  template<typename Derived_jacobian>
  bool liftJacobian(const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

 protected:

  void updateC();
  Eigen::Matrix<double, 7, 1> parameters_;  
  Eigen::Map<Eigen::Vector3d> r_;  
  Eigen::Map<Eigen::Quaterniond> q_;  
  Eigen::Matrix3d C_; 
};

typedef TransformationT<true> Transformation;

typedef TransformationT<false> TransformationCacheless;

}  
}  

#include "implementation/Transformation.hpp"

#endif 
