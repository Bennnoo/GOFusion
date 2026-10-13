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
 * @file TwoPoseExtrinsicsGraphError.hpp
 * @brief Header file for the TwoPoseExtrinsicsGraphError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_TWOPOSEEXTRINSICSGRAPHERROR_HPP_
#define INCLUDE_OKVIS_CERES_TWOPOSEEXTRINSICSGRAPHERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>
#include <ceres/loss_function.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>
#include <okvis/ceres/TwoPoseGraphError.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace okvis {

namespace ceres {

class TwoPoseExtrinsicsGraphErrorConst;

class TwoPoseExtrinsicsGraphError : public ::ceres::CostFunction

    , public TwoPoseGraphError {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  TwoPoseExtrinsicsGraphError() = delete;

  TwoPoseExtrinsicsGraphError(StateId referencePoseId, StateId otherPoseId, size_t numCams,
                              bool stayConst = true);

  virtual ~TwoPoseExtrinsicsGraphError() override = default;

  virtual bool convertToReprojectionErrors(
    std::vector<Observation> &observations,
    std::vector<KeypointIdentifier> &duplicates) override final;

  virtual bool compute() override final;

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  int residualDim() const override final {
    return num_residuals();
  }

  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const override final;

  virtual std::string typeInfo() const override final{
    return "TwoPoseExtrinsicsGraphError";
  }

  virtual double strength() const override final;

protected:

  Eigen::MatrixXd H00_;  
  Eigen::VectorXd b0_;  

  Eigen::VectorXd DeltaX_; 
  Eigen::MatrixXd J_; 

  bool errorComputationValid_;
};

class TwoPoseExtrinsicsGraphErrorConst : public ::ceres::CostFunction,
                                         public TwoPoseGraphErrorConst {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  TwoPoseExtrinsicsGraphErrorConst() = delete;

  TwoPoseExtrinsicsGraphErrorConst(
    const Eigen::VectorXd &DeltaX,
    const Eigen::MatrixXd &J,
    const kinematics::Transformation &linearisationPoint_T_S0S1,
    const std::vector<std::shared_ptr<kinematics::Transformation>> &linearisationPoints_T_SC);

  virtual ~TwoPoseExtrinsicsGraphErrorConst() override = default;

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  int residualDim() const override final {
    return num_residuals();
  }

  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::string typeInfo() const override final {
    return "TwoPoseExtrinsicsGraphErrorConst";
  }

protected:

  Eigen::VectorXd DeltaX_; 
  Eigen::MatrixXd J_; 
  kinematics::Transformation linearisationPoint_T_S0S1_; 
  std::vector<std::shared_ptr<kinematics::Transformation>> linearisationPoints_T_SC_; 
};

}  
}  

#endif 
