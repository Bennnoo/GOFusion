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
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

/**
 * @file TwoPoseGraphError.hpp
 * @brief Header file for the TwoPoseGraphError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_TWOPOSEGRAPHERROR_HPP_
#define INCLUDE_OKVIS_CERES_TWOPOSEGRAPHERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>
#include <ceres/loss_function.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace okvis {

namespace ceres {

class TwoPoseGraphErrorConst;
class TwoPoseStandardGraphError;

class TwoPoseGraphError : public ErrorInterface {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  ~TwoPoseGraphError() = default;

  void setPoseIds(StateId referencePoseId, StateId otherPoseId) {
    referencePoseId_ = referencePoseId;
    otherPoseId_ = otherPoseId;
  }

  StateId referencePoseId() const {return referencePoseId_; }

  StateId otherPoseId() const {return otherPoseId_; }

  struct Observation {
    KeypointIdentifier keypointIdentifier; 
    std::shared_ptr<ceres::PoseParameterBlock> pose; 
    std::shared_ptr<ceres::HomogeneousPointParameterBlock> hPoint; 
    std::shared_ptr<ceres::PoseParameterBlock> extrinsics; 
    std::shared_ptr<ReprojectionError2dBase> reprojectionError; 
    ::ceres::LossFunction* lossFunction; 
    bool isMarginalised; 
    bool isDuplication; 
  };

  bool addObservation(const KeypointIdentifier & keypointIdentifier,
                      const std::shared_ptr<const ReprojectionError2dBase> &reprojectionError,
                      ::ceres::LossFunction* lossFunction,
                      const std::shared_ptr<ceres::PoseParameterBlock> & pose,
                      const std::shared_ptr<ceres::HomogeneousPointParameterBlock> & hPoint,
                      const std::shared_ptr<ceres::PoseParameterBlock> & extrinsics,
                      bool isDuplication = false, double weight = 1.0);

  void getLandmarks(std::set<uint64_t> & landmarks) const;

  virtual bool convertToReprojectionErrors(std::vector<Observation> & observations,
                                   std::vector<KeypointIdentifier> & duplicates) = 0;

  int getObservations(std::vector<Observation> & observations) const {
    for (const auto &obss : observations_) {
      for (const auto &obs : obss.second) {
        observations.push_back(obs);
      }
    }
    return int(observations.size());
  }

  virtual bool compute() = 0;

  virtual double strength() const = 0;

  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const = 0;

  protected:

  bool errorComputationValid_;

  template<int SIZE>
  struct ParameterBlockInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Eigen::Matrix<double, SIZE, 1> parameters; 

    ParameterBlockInfo() {}

    ParameterBlockInfo(const std::shared_ptr<ParameterBlock> & parameterBlock, size_t idx) :
        parameterBlock(parameterBlock), idx(idx)
    {
      std::memcpy(parameters.data(), parameterBlock->parameters(), SIZE*sizeof(double));
    }
    std::shared_ptr<ParameterBlock> parameterBlock; 
    size_t idx; 
  };

  ParameterBlockInfo<7> poseParameterBlockInfos_[2]; 
  AlignedVector<ParameterBlockInfo<7>> extrinsicsParameterBlockInfos_; 
  AlignedVector<ParameterBlockInfo<4>> landmarkParameterBlockInfos_; 

  std::map<uint64_t, size_t> landmarkParameterBlockId2idx_;  
  size_t extrinsicsSize_ = 0; 
  size_t sparseSize_ = 0; 

  StateId referencePoseId_; 
  StateId otherPoseId_; 

  bool isComputed_ = false; 

  size_t numCams_ = 0;  
  bool stayConst_ = true;  

  std::map<uint64_t, std::vector<Observation>> observations_; 
  AlignedMap<uint64_t, Eigen::Vector4d> landmarks_; 

  kinematics::Transformation linearisationPoint_T_S0S1_; 
  std::vector<std::shared_ptr<kinematics::Transformation>> linearisationPoints_T_SC_; 
};

class TwoPoseStandardGraphError : public ::ceres::SizedCostFunction<
    6 ,
    7, 
    7 >, public TwoPoseGraphError {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  static const int kNumResiduals = 6;

  TwoPoseStandardGraphError() = delete;

  TwoPoseStandardGraphError(StateId referencePoseId, StateId otherPoseId, size_t numCams,
                            bool stayConst = true);

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
    return kNumResiduals;
  }

  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const override final;

  virtual std::string typeInfo() const override final{
    return "TwoPoseStandardGraphError";
  }

  virtual double strength() const override final ;

protected:

  Eigen::Matrix<double, 6, 6> H00_;  
  Eigen::Matrix<double, 6, 1> b0_;  

  Eigen::Matrix<double,6,1> DeltaX_; 
  Eigen::Matrix<double,6,6> J_; 
};

class TwoPoseGraphErrorConst : public ErrorInterface {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)
};

class TwoPoseStandardGraphErrorConst : public ::ceres::SizedCostFunction<
    6 ,
    7, 
    7 >, public TwoPoseGraphErrorConst {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  static const int kNumResiduals = 6;

  TwoPoseStandardGraphErrorConst() = delete;

  TwoPoseStandardGraphErrorConst(
      const Eigen::Matrix<double,6,1> & DeltaX, const Eigen::Matrix<double,6,6> & J,
      const kinematics::Transformation & linearisationPoint_T_S0S1) :
    DeltaX_(DeltaX), J_(J), linearisationPoint_T_S0S1_(linearisationPoint_T_S0S1)
  {}

  virtual ~TwoPoseStandardGraphErrorConst() override = default;

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  int residualDim() const override final {
    return kNumResiduals;
  }

  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::string typeInfo() const override final {
    return "TwoPoseStandardGraphErrorConst";
  }

  const Eigen::Matrix<double, 6, 1>& deltaX() const { return DeltaX_; }
  const Eigen::Matrix<double, 6, 6>& jacobian() const { return J_; }
  const kinematics::Transformation& linearisationPoint() const {
    return linearisationPoint_T_S0S1_;
  }

protected:

  Eigen::Matrix<double,6,1> DeltaX_; 
  Eigen::Matrix<double,6,6> J_; 
  kinematics::Transformation linearisationPoint_T_S0S1_; 
};

}  
}  

#endif 
