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
 * @file PseudoInverse.hpp
 * @brief Header file for the PseudoInverse class.
 * @author Stefan Leutenegger
 */

#ifndef OKVIS_PSEUDOINVERSE_HPP_
#define OKVIS_PSEUDOINVERSE_HPP_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <okvis/assert_macros.hpp>

namespace okvis {

class PseudoInverse {
public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

template<typename Derived>
static bool symm(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(), int * rank = nullptr) {
    OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
        * Eigen::VectorXd(
            (saes.eigenvalues().array() > tolerance).select(
                saes.eigenvalues().array().inverse(), 1.0/tolerance)).asDiagonal()
        * (saes.eigenvectors().transpose());

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}

template<typename Derived>
static bool symmSqrt(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(),
        int* rank = nullptr) {
    OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
        * Eigen::VectorXd(
            Eigen::VectorXd(
                (saes.eigenvalues().array() > tolerance).select(
                    saes.eigenvalues().array().inverse(), 1.0/tolerance)).array().sqrt())
            .asDiagonal();

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}

template<typename Derived>
static bool symmSqrtU(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(),
        int* rank = nullptr) {
    OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = Eigen::VectorXd(
            Eigen::VectorXd(
                (saes.eigenvalues().array() > tolerance).select(
                    saes.eigenvalues().array().inverse(), 1.0/tolerance)).array().sqrt())
            .asDiagonal() * (saes.eigenvectors().transpose());

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}
};

}

#endif 
