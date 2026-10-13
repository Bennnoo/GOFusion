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
 * @file TwoPoseExtrinsicsGraphError.cpp
 * @brief Source file for the TwoPoseExtrinsicsGraphError class.
 * @author Stefan Leutenegger
 */

#include <okvis/PseudoInverse.hpp>
#include <okvis/ceres/TwoPoseExtrinsicsGraphError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>

namespace okvis {

namespace ceres {

TwoPoseExtrinsicsGraphError::TwoPoseExtrinsicsGraphError(
  StateId referencePoseId, StateId otherPoseId, size_t numCams, bool stayConst)

{
  referencePoseId_ = referencePoseId;
  otherPoseId_ = otherPoseId;
  numCams_ = numCams;
  stayConst_ = stayConst;

  errorComputationValid_ = false;

  mutable_parameter_block_sizes()->push_back(7);

  mutable_parameter_block_sizes()->push_back(7);

  for (size_t i = 0; i < numCams_; ++i) {
    mutable_parameter_block_sizes()->push_back(7);
    extrinsicsSize_ += 6;
  }
  if (!stayConst) {

    for (size_t i = 0; i < numCams_; ++i) {
      mutable_parameter_block_sizes()->push_back(7);
      extrinsicsSize_ += 6;
    }
    extrinsicsParameterBlockInfos_.resize(2 * numCams_);
    linearisationPoints_T_SC_.resize(2 * numCams_);
  } else {
    extrinsicsParameterBlockInfos_.resize(numCams_);
    linearisationPoints_T_SC_.resize(numCams_);
  }

  set_num_residuals(6 + extrinsicsSize_);
}

bool TwoPoseExtrinsicsGraphError::compute() {
  if(isComputed_) {
    return true; 
  }

  OKVIS_ASSERT_TRUE(Exception, poseParameterBlockInfos_[0].parameterBlock,
      "reference pose not observed")
  OKVIS_ASSERT_TRUE(Exception, poseParameterBlockInfos_[1].parameterBlock,
      "other pose not observed")

  const okvis::kinematics::Transformation T_WS0 = std::static_pointer_cast<PoseParameterBlock>(
        poseParameterBlockInfos_[0].parameterBlock)->estimate();
  const okvis::kinematics::Transformation T_S0W = T_WS0.inverse();

  H00_.resize(6 + extrinsicsSize_, 6 + extrinsicsSize_);
  H00_.setZero();
  b0_.resize(6 + extrinsicsSize_);
  b0_.setZero();

  Eigen::Matrix<double, Eigen::Dynamic, 3> H01(6 + extrinsicsSize_, 3);
  H01.setZero();
  Eigen::Matrix3d H11;
  H11.setZero();
  Eigen::Vector3d b1;
  b1.setZero();

  bool relPoseSet = false;
  Eigen::MatrixXd mH(6 + extrinsicsSize_, 6 + extrinsicsSize_);
  mH.setZero();
  Eigen::VectorXd mb(6 + extrinsicsSize_);
  mb.setZero();
  for (auto &observations : observations_) {
    Eigen::MatrixXd H00(6 + extrinsicsSize_, 6 + extrinsicsSize_);
    H00.setZero();
    Eigen::VectorXd b0(6 + extrinsicsSize_);
    b0.setZero();

    Eigen::MatrixX3d H01(6 + extrinsicsSize_, 3);
    H01.setZero();
    Eigen::Matrix3d H11;
    H11.setZero();
    Eigen::Vector3d b1;
    b1.setZero();

    const uint64_t id1 = observations.first;
    const ParameterBlockInfo<4> &info1 = landmarkParameterBlockInfos_.at(
      landmarkParameterBlockId2idx_.at(id1));
    const Eigen::Vector4d hp_W(info1.parameters[0],
                               info1.parameters[1],
                               info1.parameters[2],
                               info1.parameters[3]);

    const Eigen::Vector4d hp_S0 = T_S0W * hp_W;
    landmarks_[id1] = hp_S0;

    const double minDist = hp_S0[2] / hp_S0[3];

    for (auto &observation : observations.second) {
      const uint64_t id0 = observation.keypointIdentifier.frameId;
      const size_t ci = observation.keypointIdentifier.cameraIndex;
      const size_t poseIdx = (id0 == referencePoseId_.value()) ? 0 : 1;
      const size_t offset = (stayConst_) ? 0 : poseIdx * numCams_;
      const ParameterBlockInfo<7> &info0 = poseParameterBlockInfos_[poseIdx];
      const ParameterBlockInfo<7> &info2 = extrinsicsParameterBlockInfos_[offset + ci];

      const okvis::kinematics::Transformation T_SCi(Eigen::Vector3d(info2.parameters[0],
                                                                    info2.parameters[1],
                                                                    info2.parameters[2]),
                                                    Eigen::Quaterniond(info2.parameters[6],
                                                                       info2.parameters[3],
                                                                       info2.parameters[4],
                                                                       info2.parameters[5]));

      const bool isReference = id0 == referencePoseId_.value();

      Eigen::Vector2d residual;
      Eigen::Matrix<double, 2, 7, Eigen::RowMajor> jacobian0;
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> minimalJacobian0;
      Eigen::Matrix<double, 2, 4, Eigen::RowMajor> jacobian1;
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> minimalJacobian1;
      Eigen::Matrix<double, 2, 7, Eigen::RowMajor> jacobian2;
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> minimalJacobian2;
      double *jacobians[3];
      double *minimalJacobians[3];
      if (isReference) {
        jacobians[0] = nullptr;
        minimalJacobians[0] = nullptr;
      } else {
        jacobians[0] = jacobian0.data();
        minimalJacobians[0] = minimalJacobian0.data();
      }
      jacobians[1] = jacobian1.data();
      jacobians[2] = jacobian2.data();
      minimalJacobians[1] = minimalJacobian1.data();
      minimalJacobians[2] = minimalJacobian2.data();

      okvis::kinematics::Transformation T_S0S; 
      if (!isReference) {

        const okvis::kinematics::Transformation T_WS(Eigen::Vector3d(info0.parameters[0],
                                                                     info0.parameters[1],
                                                                     info0.parameters[2]),
                                                     Eigen::Quaterniond(info0.parameters[6],
                                                                        info0.parameters[3],
                                                                        info0.parameters[4],
                                                                        info0.parameters[5]));
        T_S0S = T_S0W * T_WS;
      }

      if (!isReference && !relPoseSet) {
        linearisationPoint_T_S0S1_ = T_S0S;
        relPoseSet = true;
      }

      const double* params[3];
      PoseParameterBlock pose(T_S0S, id0, okvis::Time(0));
      params[0] = pose.parameters();
      params[1] = hp_S0.data();
      params[2] = info2.parameters.data();

      observation.reprojectionError->EvaluateWithMinimalJacobians(
                  params, residual.data(), jacobians, minimalJacobians);

      if(residual.norm()>3.0) {
        continue;
      }

      {

        const ::ceres::LossFunction* lossFunction = observation.lossFunction;
        if (lossFunction) {

          const double sq_norm = residual.transpose() * residual;
          double rho[3];
          lossFunction->Evaluate(sq_norm, rho);
          const double sqrt_rho1 = sqrt(rho[1]);
          double residual_scaling;
          double alpha_sq_norm;
          if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling = sqrt_rho1;
            alpha_sq_norm = 0.0;

          } else {

            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

            const double alpha = 1.0 - sqrt(D);

            residual_scaling = sqrt_rho1 / (1 - alpha);
            alpha_sq_norm = alpha / sq_norm;
          }

          minimalJacobian0 = sqrt_rho1 * (minimalJacobian0 - alpha_sq_norm * residual
                                          * (residual.transpose() * minimalJacobian0));
          minimalJacobian1 = sqrt_rho1 * (minimalJacobian1 - alpha_sq_norm * residual
                                          * (residual.transpose() * minimalJacobian1));
          minimalJacobian2 = sqrt_rho1 * (minimalJacobian2 - alpha_sq_norm * residual
                                          * (residual.transpose() * minimalJacobian2));

          residual *= residual_scaling;
        }

        const int start2 = int(info2.idx);

        if(!isReference) {
          H00.topLeftCorner<6,6>() += minimalJacobian0.transpose()*minimalJacobian0;
          b0.head<6>() -= minimalJacobian0.transpose()*residual;
          H01.block<6,3>(0,0) += minimalJacobian0.transpose()*minimalJacobian1;

          H00.block<6, 6>(0, start2) += minimalJacobian0.transpose() * minimalJacobian2;
          H00.block<6, 6>(start2, 0) += minimalJacobian2.transpose() * minimalJacobian0;
        }

        H01.block<6,3>(start2, 0) += minimalJacobian2.transpose()*minimalJacobian1;

        H00.block<6,6>(start2, start2) += minimalJacobian2.transpose()*minimalJacobian2;
        b0.segment<6>(start2) -= minimalJacobian2.transpose()*residual;

        H11 += minimalJacobian1.transpose()*minimalJacobian1;
        b1 -= minimalJacobian1.transpose()*residual;
      }

      observation.isMarginalised = true;
    }

    const Eigen::MatrixX3d W = H01;
    const Eigen::Matrix3d V = H11;
    Eigen::Matrix3d V_inv_sqrt;
    int rank;
    PseudoInverse::symmSqrt(V, V_inv_sqrt, 1.0e-7, &rank);
    if (rank < 3 && minDist < 2.99) {

    } else {
      H00_ += H00;
      b0_ += b0;
      const Eigen::MatrixX3d M = W * V_inv_sqrt;
      mH += M * M.transpose();
      mb += M * (V_inv_sqrt.transpose() * b1);
    }
  }
  H00_ -= mH;
  b0_ -= mb;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H00_);
  const double tolerance = 1.0e-8 * double(H00_.cols()) * saes.eigenvalues().array().maxCoeff();
  const Eigen::VectorXd D_sqrt = Eigen::VectorXd((saes.eigenvalues().array() > tolerance).select(
                saes.eigenvalues().array().sqrt(), 0));
  const Eigen::VectorXd D_inv_sqrt =
      Eigen::VectorXd((saes.eigenvalues().array() > tolerance).select(
                        saes.eigenvalues().array().inverse().sqrt(), 0));

  J_ = D_sqrt.asDiagonal() * saes.eigenvectors().transpose();
  const Eigen::MatrixXd M = saes.eigenvectors() * D_inv_sqrt.asDiagonal();
  DeltaX_ = -M * (M.transpose() * b0_);

  sparseSize_ = 0;

  landmarkParameterBlockId2idx_.clear();

  isComputed_ = true;

  return true;
}

bool TwoPoseExtrinsicsGraphError::convertToReprojectionErrors(std::vector<Observation> & observations,
                                                 std::vector<KeypointIdentifier> & duplicates) {

  const okvis::kinematics::Transformation T_WS0 = std::static_pointer_cast<PoseParameterBlock>(
      poseParameterBlockInfos_[0].parameterBlock)->estimate();

  for (auto &obss : observations_) {
    for (auto &observation : obss.second) {

      if (!observation.isMarginalised) {
        continue; 
      }

      if (observation.isDuplication) {

        duplicates.push_back(observation.keypointIdentifier);

      }

      const uint64_t landmarkId = observation.hPoint->id();
      std::shared_ptr<HomogeneousPointParameterBlock> landmark;

      Eigen::Vector4d hp_W = T_WS0 * landmarks_.at(landmarkId);
      observation.hPoint.reset(
        new ceres::HomogeneousPointParameterBlock(hp_W,
                                                  landmarkId,
                                                  observation.hPoint->initialized()));

        observations.push_back(observation);
    }
  }

  landmarkParameterBlockId2idx_.clear();
  landmarkParameterBlockInfos_.clear();
  observations_.clear();
  landmarks_.clear();
  for (size_t i = 0; i < extrinsicsParameterBlockInfos_.size(); ++i) {
    extrinsicsParameterBlockInfos_[i].parameterBlock.reset();
    linearisationPoints_T_SC_[i].reset();
  }

  J_.setZero();
  H00_.setZero();
  b0_.setZero();
  sparseSize_ = 0;
  return true;
}

bool TwoPoseExtrinsicsGraphError::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool TwoPoseExtrinsicsGraphError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                  double* residuals, double** jacobians,
                                  double** jacobiansMinimal) const {
  if (!isComputed_) {
    return false;
  }

  Eigen::Map<Eigen::VectorXd> error_weighted(residuals, 6 + extrinsicsSize_);

  const size_t refIdx = 0;
  const okvis::kinematics::Transformation T_WS0(
        Eigen::Vector3d(parameters[refIdx][0],parameters[refIdx][1],parameters[refIdx][2]),
        Eigen::Quaterniond(parameters[refIdx][6],parameters[refIdx][3],parameters[refIdx][4],
      parameters[refIdx][5]).normalized());
  const okvis::kinematics::Transformation T_S0W = T_WS0.inverse();

  Eigen::VectorXd DeltaX_linearisationPoint(6 + extrinsicsSize_);
  DeltaX_linearisationPoint.setZero();

  const size_t idx = 1;
  const size_t startIdx = 0;

  const okvis::kinematics::Transformation T_WSi(
        Eigen::Vector3d(parameters[idx][0],parameters[idx][1],parameters[idx][2]),
        Eigen::Quaterniond(parameters[idx][6],parameters[idx][3],parameters[idx][4],
      parameters[idx][5]).normalized());
  const okvis::kinematics::Transformation T_S0Si = T_S0W * T_WSi;
  DeltaX_linearisationPoint.segment<3>(startIdx) =
      T_S0Si.r() - linearisationPoint_T_S0S1_.r();
  DeltaX_linearisationPoint.segment<3>(startIdx + 3) =
      2.0*(T_S0Si.q() * linearisationPoint_T_S0S1_.q().inverse()).coeffs().head<3>();

  int i = 0;
  for (const auto &extrinsicsInfo : extrinsicsParameterBlockInfos_) {
    const okvis::kinematics::Transformation T_SCi(Eigen::Vector3d(parameters[2 + i][0],
                                                                  parameters[2 + i][1],
                                                                  parameters[2 + i][2]),
                                                  Eigen::Quaterniond(parameters[2 + i][6],
                                                                     parameters[2 + i][3],
                                                                     parameters[2 + i][4],
                                                                     parameters[2 + i][5])
                                                    .normalized());
    if (linearisationPoints_T_SC_[i]) {
      const double *parLin = extrinsicsInfo.parameters.data();
      const okvis::kinematics::Transformation
        T_SCi_lin(Eigen::Vector3d(parLin[0], parLin[1], parLin[2]),
                  Eigen::Quaterniond(parLin[6], parLin[3], parLin[4], parLin[5]).normalized());
      DeltaX_linearisationPoint.segment<3>(extrinsicsInfo.idx) =
        T_SCi.r() - linearisationPoints_T_SC_[i]->r();
      DeltaX_linearisationPoint.segment<3>(extrinsicsInfo.idx + 3)
        = 2.0 * (T_SCi.q() * linearisationPoints_T_SC_[i]->q().inverse()).coeffs().head<3>();
    }
    i++;
  }

  Eigen::VectorXd error(6 + extrinsicsSize_);
  error = DeltaX_ + DeltaX_linearisationPoint;

  error_weighted = J_ * error;

  if(jacobians || jacobiansMinimal) {

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> JerrRef =
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

    const size_t idx = 1;
    if(jacobians[idx]) {

      const size_t startIdx = 0;

      Eigen::MatrixXd Jmin(6+extrinsicsSize_,6);

      const okvis::kinematics::Transformation T_WS(Eigen::Vector3d(parameters[idx][0],
                                                                   parameters[idx][1],
                                                                   parameters[idx][2]),
                                                   Eigen::Quaterniond(parameters[idx][6],
                                                                      parameters[idx][3],
                                                                      parameters[idx][4],
                                                                      parameters[idx][5]));
      Eigen::Matrix<double,6,6> Jerr = Eigen::Matrix<double,6,6>::Zero();
      Jerr.topLeftCorner<3,3>() = T_S0W.C();
      Jerr.bottomRightCorner<3, 3>() = (okvis::kinematics::plus(T_WS0.q().inverse())
                                        * okvis::kinematics::oplus(
                                          T_WS.q() * linearisationPoint_T_S0S1_.q().inverse()))
                                         .topLeftCorner<3, 3>();
      Jmin = J_.block(0,startIdx,6+extrinsicsSize_,6)*Jerr;

      if(jacobians[refIdx]) {
        JerrRef.block<3,3>(startIdx, 0) = -T_S0W.C();
        JerrRef.block<3,3>(startIdx, 0+3) =
          T_S0W.C()*okvis::kinematics::crossMx(T_WS.r()-T_WS0.r());
        JerrRef.block<3,3>(startIdx+3, 0+3) = -Jerr.bottomRightCorner<3,3>();
      }

      if(jacobiansMinimal){
        if(jacobiansMinimal[idx]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>>
            JminMapped(jacobiansMinimal[idx], 6+extrinsicsSize_, 6);
          JminMapped = Jmin;
        }
      }

      if(jacobians[idx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[idx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
          J(jacobians[idx], 6+extrinsicsSize_, 7);
        J = Jmin * J_lift;
      }
    }

    if(jacobians[refIdx]) {

      const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> Jmin =
        J_.block(0, 0, 6+extrinsicsSize_,6) * JerrRef;

      if(jacobiansMinimal) {
        if(jacobiansMinimal[refIdx]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> JminMapped(
                jacobiansMinimal[refIdx], 6+extrinsicsSize_, 6);
          JminMapped = Jmin;
        }
      }

      if(jacobians[refIdx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[refIdx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
          J(jacobians[refIdx], 6+extrinsicsSize_, 7);
        J = Jmin * J_lift;
      }
    }

    for (size_t i=0; i<extrinsicsParameterBlockInfos_.size(); ++i) {
      int extrinsicsIdx = i+2;
      const okvis::kinematics::Transformation T_SCi(Eigen::Vector3d(parameters[extrinsicsIdx][0],
                                                                    parameters[extrinsicsIdx][1],
                                                                    parameters[extrinsicsIdx][2]),
                                                    Eigen::Quaterniond(parameters[extrinsicsIdx][6],
                                                                       parameters[extrinsicsIdx][3],
                                                                       parameters[extrinsicsIdx][4],
                                                                       parameters[extrinsicsIdx][5])
                                                      .normalized());

      Eigen::Matrix<double, 6, 6> Jerrex = Eigen::Matrix<double, 6, 6>::Zero();
      if (linearisationPoints_T_SC_[i]) {
        const okvis::kinematics::Transformation & T_SCi_lin
          = *linearisationPoints_T_SC_[i];
        Jerrex.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
        Jerrex.bottomRightCorner<3, 3>()
          = (okvis::kinematics::oplus(T_SCi.q() * T_SCi_lin.q().inverse())).topLeftCorner<3, 3>();
      }

      if (jacobians[extrinsicsIdx]) {

        const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> Jmin
          = J_.block(0, 6+i*6, 6+extrinsicsSize_, 6) * Jerrex;

        if(jacobiansMinimal) {
          if(jacobiansMinimal[extrinsicsIdx]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> JminMapped(
              jacobiansMinimal[extrinsicsIdx], 6+extrinsicsSize_, 6);
            JminMapped = Jmin;
          }
        }

        if(jacobians[extrinsicsIdx]) {

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[extrinsicsIdx], J_lift.data());

          Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic, 7, Eigen::RowMajor>> J(
            jacobians[extrinsicsIdx], J_.rows(), 7);
          J = Jmin * J_lift;
        }
      }
    }
  }

  return true;
}

std::shared_ptr<TwoPoseGraphErrorConst> TwoPoseExtrinsicsGraphError::cloneTwoPoseGraphErrorConst()
  const
{
  return std::shared_ptr<TwoPoseGraphErrorConst>(
        new TwoPoseExtrinsicsGraphErrorConst(
          DeltaX_, J_, linearisationPoint_T_S0S1_, linearisationPoints_T_SC_));
}

double TwoPoseExtrinsicsGraphError::strength() const {
  if(isComputed_) {
    Eigen::Matrix<double, 6, 6> H00inv;
    const Eigen::Matrix<double, 6, 6> H00 = H00_.topLeftCorner<6, 6>();
    int rank;
    PseudoInverse::symm(H00, H00inv, 1.0e-6, &rank);
    if(rank<6) {
      return 0.0;
    }
    return 1.0/sqrt(H00inv.norm()); 
  }
  return 0.0;
}

TwoPoseExtrinsicsGraphErrorConst::TwoPoseExtrinsicsGraphErrorConst(
  const Eigen::VectorXd &DeltaX,
  const Eigen::MatrixXd &J,
  const kinematics::Transformation &linearisationPoint_T_S0S1,
  const std::vector<std::shared_ptr<kinematics::Transformation>> &linearisationPoints_T_SC)
    : DeltaX_(DeltaX)
    , J_(J)
    , linearisationPoint_T_S0S1_(linearisationPoint_T_S0S1)
    , linearisationPoints_T_SC_(linearisationPoints_T_SC)
{

  const size_t numCams = (DeltaX.size() - 6) / 6;
  mutable_parameter_block_sizes()->push_back(7); 
  mutable_parameter_block_sizes()->push_back(7); 
  for (size_t i = 0; i < numCams; ++i) {
    mutable_parameter_block_sizes()->push_back(7);
  }
  set_num_residuals(DeltaX.size());
}

bool TwoPoseExtrinsicsGraphErrorConst::Evaluate(const double *const *parameters,
                                                double *residuals,
                                                double **jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool TwoPoseExtrinsicsGraphErrorConst::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                                    double *residuals,
                                                                    double **jacobians,
                                                                    double **jacobiansMinimal) const
{
  const int extrinsicsSize = DeltaX_.size() - 6;

  Eigen::Map<Eigen::VectorXd> error_weighted(residuals, 6 + extrinsicsSize);

  const size_t refIdx = 0;
  const okvis::kinematics::Transformation T_WS0(Eigen::Vector3d(parameters[refIdx][0],
                                                                parameters[refIdx][1],
                                                                parameters[refIdx][2]),
                                                Eigen::Quaterniond(parameters[refIdx][6],
                                                                   parameters[refIdx][3],
                                                                   parameters[refIdx][4],
                                                                   parameters[refIdx][5])
                                                  .normalized());
  const okvis::kinematics::Transformation T_S0W = T_WS0.inverse();

  Eigen::VectorXd DeltaX_linearisationPoint(6 + extrinsicsSize);
  DeltaX_linearisationPoint.setZero();

  const size_t idx = 1;
  const size_t startIdx = 0;

  const okvis::kinematics::Transformation T_WSi(Eigen::Vector3d(parameters[idx][0],
                                                                parameters[idx][1],
                                                                parameters[idx][2]),
                                                Eigen::Quaterniond(parameters[idx][6],
                                                                   parameters[idx][3],
                                                                   parameters[idx][4],
                                                                   parameters[idx][5])
                                                  .normalized());
  const okvis::kinematics::Transformation T_S0Si = T_S0W * T_WSi;
  DeltaX_linearisationPoint.segment<3>(startIdx) = T_S0Si.r() - linearisationPoint_T_S0S1_.r();
  DeltaX_linearisationPoint.segment<3>(startIdx + 3)
    = 2.0 * (T_S0Si.q() * linearisationPoint_T_S0S1_.q().inverse()).coeffs().head<3>();

  int i = 0;
  for (const auto &T_SCi_lin : linearisationPoints_T_SC_) {
    const okvis::kinematics::Transformation T_SCi(Eigen::Vector3d(parameters[2 + i][0],
                                                                  parameters[2 + i][1],
                                                                  parameters[2 + i][2]),
                                                  Eigen::Quaterniond(parameters[2 + i][6],
                                                                     parameters[2 + i][3],
                                                                     parameters[2 + i][4],
                                                                     parameters[2 + i][5])
                                                    .normalized());
    if (T_SCi_lin) {
      DeltaX_linearisationPoint.segment<3>(6 + i * 6) = T_SCi.r() - T_SCi_lin->r();
      DeltaX_linearisationPoint.segment<3>(6 + i * 6 + 3)
        = 2.0 * (T_SCi.q() * T_SCi_lin->q().inverse()).coeffs().head<3>();
    }
    i++;
  }

  Eigen::VectorXd error(6 + extrinsicsSize);
  error = DeltaX_ + DeltaX_linearisationPoint;

  error_weighted = J_ * error;

  if (jacobians || jacobiansMinimal) {

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> JerrRef
      = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

    const size_t idx = 1;
    if (jacobians[idx]) {
      const size_t startIdx = 0;

      Eigen::MatrixXd Jmin(6 + extrinsicsSize, 6);

      const okvis::kinematics::Transformation T_WS(Eigen::Vector3d(parameters[idx][0],
                                                                   parameters[idx][1],
                                                                   parameters[idx][2]),
                                                   Eigen::Quaterniond(parameters[idx][6],
                                                                      parameters[idx][3],
                                                                      parameters[idx][4],
                                                                      parameters[idx][5]));
      Eigen::Matrix<double, 6, 6> Jerr = Eigen::Matrix<double, 6, 6>::Zero();
      Jerr.topLeftCorner<3, 3>() = T_S0W.C();
      Jerr.bottomRightCorner<3, 3>() = (okvis::kinematics::plus(T_WS0.q().inverse())
                                        * okvis::kinematics::oplus(
                                          T_WS.q() * linearisationPoint_T_S0S1_.q().inverse()))
                                         .topLeftCorner<3, 3>();
      Jmin = J_.block(0, startIdx, 6 + extrinsicsSize, 6) * Jerr;

      if (jacobians[refIdx]) {
        JerrRef.block<3, 3>(startIdx, 0) = -T_S0W.C();
        JerrRef.block<3, 3>(startIdx, 0 + 3) = T_S0W.C()
                                               * okvis::kinematics::crossMx(T_WS.r() - T_WS0.r());
        JerrRef.block<3, 3>(startIdx + 3, 0 + 3) = -Jerr.bottomRightCorner<3, 3>();
      }

      if (jacobiansMinimal) {
        if (jacobiansMinimal[idx]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>>
            JminMapped(jacobiansMinimal[idx], 6 + extrinsicsSize, 6);
          JminMapped = Jmin;
        }
      }

      if (jacobians[idx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[idx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> J(jacobians[idx],
                                                                                6 + extrinsicsSize,
                                                                                7);
        J = Jmin * J_lift;
      }
    }

    if (jacobians[refIdx]) {

      const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> Jmin
        = J_.block(0, 0, 6 + extrinsicsSize, 6) * JerrRef;

      if (jacobiansMinimal) {
        if (jacobiansMinimal[refIdx]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>>
            JminMapped(jacobiansMinimal[refIdx], 6 + extrinsicsSize, 6);
          JminMapped = Jmin;
        }
      }

      if (jacobians[refIdx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[refIdx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> J(jacobians[refIdx],
                                                                                6 + extrinsicsSize,
                                                                                7);
        J = Jmin * J_lift;
      }
    }

    for (size_t i = 0; i < linearisationPoints_T_SC_.size(); ++i) {
      if (jacobians[i + 2]) {

        const okvis::kinematics::Transformation T_SCi(Eigen::Vector3d(parameters[2 + i][0],
                                                                      parameters[2 + i][1],
                                                                      parameters[2 + i][2]),
                                                      Eigen::Quaterniond(parameters[2 + i][6],
                                                                         parameters[2 + i][3],
                                                                         parameters[2 + i][4],
                                                                         parameters[2 + i][5])
                                                        .normalized());

        Eigen::Matrix<double, 6, 6> Jerrex = Eigen::Matrix<double, 6, 6>::Zero();
        if (linearisationPoints_T_SC_[i]) {
          Jerrex.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
          Jerrex.bottomRightCorner<3, 3>() = (okvis::kinematics::oplus(
                                                T_SCi.q()
                                                * linearisationPoints_T_SC_[i]->q().inverse()))
                                               .topLeftCorner<3, 3>();
        }

        const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> Jmin
          = J_.block(0, 6+i*6, 6+extrinsicsSize, 6) * Jerrex;

        if(jacobiansMinimal) {
          if(jacobiansMinimal[i+2]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> JminMapped(
              jacobiansMinimal[i+2], 6+extrinsicsSize, 6);
            JminMapped = Jmin;
          }
        }

        if(jacobians[i+2]) {

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[i+2], J_lift.data());

          Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic, 7, Eigen::RowMajor>> J(
            jacobians[i+2], J_.rows(), 7);
          J = Jmin * J_lift;
        }
      }
    }
  }

  return true;
}

} 
} 
