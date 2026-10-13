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
 * @file TwoPoseGraphError.cpp
 * @brief Source file for the TwoPoseGraphError class.
 * @author Stefan Leutenegger
 */

#include <okvis/PseudoInverse.hpp>
#include <okvis/ceres/TwoPoseGraphError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>

namespace okvis {

namespace ceres {

TwoPoseStandardGraphError::TwoPoseStandardGraphError(StateId referencePoseId, StateId otherPoseId,
                                                     size_t numCams,
                                                     bool stayConst)
{
  referencePoseId_ = referencePoseId;
  otherPoseId_ = otherPoseId;
  numCams_ = numCams;
  stayConst_ = stayConst;
  errorComputationValid_ = false;
  H00_.setZero();
  b0_.setZero();
  J_.setZero();
  DeltaX_.setZero();

  for (size_t i = 0; i < numCams_; ++i) {
    extrinsicsSize_ += 6;
  }
  if(!stayConst) {

    for(size_t i=0; i<numCams_; ++i) {
      extrinsicsSize_ += 6;
    }
    extrinsicsParameterBlockInfos_.resize(2 * numCams_);
    linearisationPoints_T_SC_.resize(2 * numCams_);
  } else {
    extrinsicsParameterBlockInfos_.resize(numCams_);
    linearisationPoints_T_SC_.resize(numCams_);
  }
}

bool TwoPoseGraphError::addObservation(
  const KeypointIdentifier &keypointIdentifier,
  const std::shared_ptr<const ReprojectionError2dBase> &reprojectionError,
  ::ceres::LossFunction *lossFunction,
  const std::shared_ptr<ceres::PoseParameterBlock> &pose,
  const std::shared_ptr<ceres::HomogeneousPointParameterBlock> &hPoint,
  const std::shared_ptr<ceres::PoseParameterBlock> &extrinsics,
  bool isDuplication,
  double weight)
{
  errorComputationValid_ = false; 

  OKVIS_ASSERT_TRUE_DBG(Exception, referencePoseId_.isInitialised(), "reference pose not set")
  OKVIS_ASSERT_TRUE_DBG(Exception, otherPoseId_.isInitialised(), "other pose not set")

  const size_t ci = reprojectionError->cameraId();
  if (weight != 1.0) { 

    auto reprojectionErrorClone = reprojectionError->clone();
    reprojectionErrorClone->setInformation(weight * reprojectionErrorClone->information());
    Observation obs{keypointIdentifier,
                    pose,
                    hPoint,
                    extrinsics,
                    reprojectionErrorClone,
                    lossFunction,
                    false,
                    isDuplication};
    observations_[hPoint->id()].push_back(obs);
  } else {
    if (isDuplication) {

      auto reprojectionErrorClone = reprojectionError->clone();
      reprojectionErrorClone->setInformation(reprojectionErrorClone->information());
      Observation obs{keypointIdentifier,
                      pose,
                      hPoint,
                      extrinsics,
                      reprojectionErrorClone,
                      lossFunction,
                      false,
                      true};
      observations_[hPoint->id()].push_back(obs);
    } else {

      auto reprojectionErrorClone = reprojectionError->clone();
      Observation obs{keypointIdentifier,
                      pose,
                      hPoint,
                      extrinsics,
                      reprojectionErrorClone,
                      lossFunction,
                      false,
                      false};
      observations_[hPoint->id()].push_back(obs);
    }
  }

  size_t poseIdx = 0;
  if (pose->id() == referencePoseId_.value()) {
    if (!poseParameterBlockInfos_[0].parameterBlock) {
      poseParameterBlockInfos_[0] = ParameterBlockInfo<7>(pose, 0);
      poseIdx = 0;
    }
  } else if (pose->id() == otherPoseId_.value()) {
    if (!poseParameterBlockInfos_[1].parameterBlock) {
      poseParameterBlockInfos_[1] = ParameterBlockInfo<7>(pose, 0);
      poseIdx = 1;
    }
  } else {
    OKVIS_THROW(Exception, "pose not registered")
    return false;
  }

  if (landmarkParameterBlockId2idx_.find(hPoint->id()) == landmarkParameterBlockId2idx_.end()) {

    landmarkParameterBlockInfos_.push_back(ParameterBlockInfo<4>(hPoint, sparseSize_));
    landmarkParameterBlockId2idx_[hPoint->id()] = landmarkParameterBlockInfos_.size() - 1;
    sparseSize_ += 3;

  }

  const size_t offset = (stayConst_) ? 0 : poseIdx * numCams_;
  if (!extrinsicsParameterBlockInfos_[offset + ci].parameterBlock) {
    extrinsicsParameterBlockInfos_[offset + ci]
      = ParameterBlockInfo<7>(extrinsics, 6 + offset + ci * 6); 
    linearisationPoints_T_SC_[offset + ci].reset(
      new kinematics::Transformation(extrinsics->estimate()));
  }

  return true;
}

bool TwoPoseStandardGraphError::compute() {
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

  bool relPoseSet = false;
  Eigen::Matrix<double, 6, 6> mH = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> mb = Eigen::Matrix<double, 6, 1>::Zero();
  for (auto &observations : observations_) {
    Eigen::Matrix<double, 6, 6> H00;
    H00.setZero();
    Eigen::Matrix<double, 6, 1> b0;
    b0.setZero();

    Eigen::Matrix<double, 6, 3> H01;
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
      jacobians[2] = nullptr; 
      minimalJacobians[1] = minimalJacobian1.data();
      minimalJacobians[2] = nullptr; 

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

      const double *params[3];
      PoseParameterBlock pose(T_S0S, id0, okvis::Time(0));
      params[0] = pose.parameters();
      params[1] = hp_S0.data();
      params[2] = info2.parameters.data();

      observation.reprojectionError->EvaluateWithMinimalJacobians(params,
                                                                  residual.data(),
                                                                  jacobians,
                                                                  minimalJacobians);

      if (residual.norm() > 3.0) {
        continue;
      }

      {

        const ::ceres::LossFunction *lossFunction = observation.lossFunction;
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

          minimalJacobian0 = sqrt_rho1
                             * (minimalJacobian0
                                - alpha_sq_norm * residual
                                    * (residual.transpose() * minimalJacobian0));
          minimalJacobian1 = sqrt_rho1
                             * (minimalJacobian1
                                - alpha_sq_norm * residual
                                    * (residual.transpose() * minimalJacobian1));

          residual *= residual_scaling;
        }

        if (!isReference) {
          H00 += minimalJacobian0.transpose() * minimalJacobian0;
          b0 -= minimalJacobian0.transpose() * residual;
          H01 += minimalJacobian0.transpose() * minimalJacobian1;
        }

        H11 += minimalJacobian1.transpose() * minimalJacobian1;
        b1 -= minimalJacobian1.transpose() * residual;
      }

      observation.isMarginalised = true;
    }

    const Eigen::Matrix<double, 6, 3> W = H01;
    const Eigen::Matrix3d V = H11;
    Eigen::Matrix3d V_inv_sqrt;
    int rank;
    PseudoInverse::symmSqrt(V, V_inv_sqrt, 1.0e-7, &rank);
    if (rank < 3 && minDist < 2.99) {

    } else {
      H00_ += H00;
      b0_ += b0;
      const Eigen::Matrix<double, 6, 3> M = W * V_inv_sqrt;
      mH += M * M.transpose();
      mb += M * (V_inv_sqrt.transpose() * b1);
    }
  }
  H00_ -= mH;
  b0_ -= mb;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> saes(H00_);
  const double tol = 1.0e-8 * double(H00_.cols()) * saes.eigenvalues().array().maxCoeff();
  const Eigen::Matrix<double, 6, 1> D_sqrt = Eigen::Matrix<double, 6, 1>(
    (saes.eigenvalues().array() > tol).select(saes.eigenvalues().array().sqrt(), 0));
  const Eigen::Matrix<double, 6, 1> D_inv_sqrt = Eigen::Matrix<double, 6, 1>(
    (saes.eigenvalues().array() > tol).select(saes.eigenvalues().array().inverse().sqrt(), 0));

  J_ = D_sqrt.asDiagonal() * saes.eigenvectors().transpose();
  const Eigen::Matrix<double,6,6> M = saes.eigenvectors() * D_inv_sqrt.asDiagonal();
  DeltaX_ = -M * (M.transpose() * b0_);

  sparseSize_ = 0;

  landmarkParameterBlockId2idx_.clear();

  isComputed_ = true;

  return true;
}

bool TwoPoseStandardGraphError::convertToReprojectionErrors(std::vector<Observation> & observations,
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
\

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

void TwoPoseGraphError::getLandmarks(std::set<uint64_t> & landmarks) const {
  for(auto it=landmarkParameterBlockId2idx_.begin();
      it!=landmarkParameterBlockId2idx_.end(); ++it) {
    landmarks.insert(it->first);
  }
}

bool TwoPoseStandardGraphError::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool TwoPoseStandardGraphError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                  double* residuals, double** jacobians,
                                  double** jacobiansMinimal) const {
  if (!isComputed_) {
    return false;
  }

  Eigen::Map<Eigen::Matrix<double,6,1>> error_weighted(residuals, 6);

  const size_t refIdx = 0;
  const okvis::kinematics::Transformation T_WS0(
        Eigen::Vector3d(parameters[refIdx][0],parameters[refIdx][1],parameters[refIdx][2]),
        Eigen::Quaterniond(parameters[refIdx][6],parameters[refIdx][3],parameters[refIdx][4],
      parameters[refIdx][5]).normalized());
  const okvis::kinematics::Transformation T_S0W = T_WS0.inverse();

  Eigen::Matrix<double,6,1> DeltaX_linearisationPoint = Eigen::Matrix<double,6,1>::Zero();

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

  const Eigen::Matrix<double,6,1> error = DeltaX_ + DeltaX_linearisationPoint;

  error_weighted = J_ * error;

  if(jacobians || jacobiansMinimal) {

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> JerrRef =
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

    const size_t idx = 1;
    if(jacobians[idx]) {

      const size_t startIdx = 0;

      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jmin;

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
      Jmin = J_.block<6,6>(0,startIdx)*Jerr;

      if(jacobians[refIdx]) {
        JerrRef.block<3,3>(startIdx, 0) = -T_S0W.C();
        JerrRef.block<3, 3>(startIdx, 0 + 3) = T_S0W.C()
                                               * okvis::kinematics::crossMx(T_WS.r() - T_WS0.r());
        JerrRef.block<3,3>(startIdx+3, 0+3) = -Jerr.bottomRightCorner<3,3>();
      }

      if(jacobiansMinimal){
        if(jacobiansMinimal[idx]) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> JminMapped(
                jacobiansMinimal[idx]);
          JminMapped = Jmin;
        }
      }

      if(jacobians[idx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[idx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[idx]);
        J = Jmin * J_lift;
      }
    }

    if(jacobians[refIdx]) {

      const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jmin =  J_ * JerrRef;

      if(jacobiansMinimal) {
        if(jacobiansMinimal[refIdx]) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> JminMapped(
            jacobiansMinimal[refIdx]);
          JminMapped = Jmin;
        }
      }

      if(jacobians[refIdx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[refIdx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[refIdx]);
        J = Jmin * J_lift;
      }
    }
  }

  return true;
}

std::shared_ptr<TwoPoseGraphErrorConst>
TwoPoseStandardGraphError::cloneTwoPoseGraphErrorConst() const {
  return std::shared_ptr<TwoPoseGraphErrorConst>(
    new TwoPoseStandardGraphErrorConst(DeltaX_, J_, linearisationPoint_T_S0S1_));
}

double TwoPoseStandardGraphError::strength() const {
  if(isComputed_) {
    Eigen::Matrix<double, 6, 6> H00inv;
    int rank;
    PseudoInverse::symm(H00_, H00inv, 1.0e-6, &rank);
    if(rank<6) {
      return 0.0;
    }
    return 1.0/sqrt(H00inv.norm()); 
  }
  return 0.0;
}

bool TwoPoseStandardGraphErrorConst::Evaluate(
    const double * const *parameters, double *residuals, double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool TwoPoseStandardGraphErrorConst::EvaluateWithMinimalJacobians(
    const double * const *parameters, double *residuals, double **jacobians,
    double **jacobiansMinimal) const
{

  Eigen::Map<Eigen::Matrix<double,6,1>> error_weighted(residuals, 6);

  const size_t refIdx = 0;
  const okvis::kinematics::Transformation T_WS0(
        Eigen::Vector3d(parameters[refIdx][0],parameters[refIdx][1],parameters[refIdx][2]),
        Eigen::Quaterniond(parameters[refIdx][6],parameters[refIdx][3],parameters[refIdx][4],
      parameters[refIdx][5]).normalized());
  const okvis::kinematics::Transformation T_S0W = T_WS0.inverse();

  Eigen::Matrix<double,6,1> DeltaX_linearisationPoint = Eigen::Matrix<double,6,1>::Zero();

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
      2*(T_S0Si.q() * linearisationPoint_T_S0S1_.q().inverse()).coeffs().head<3>();

  const Eigen::Matrix<double,6,1> error = DeltaX_ + DeltaX_linearisationPoint;

  error_weighted = J_ * error;

  if(jacobians || jacobiansMinimal) {

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> JerrRef =
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

    const size_t idx = 1;
    if(jacobians[idx]) {

      const size_t startIdx = 0;

      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jmin;

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
      Jmin = J_.block<6,6>(0,startIdx)*Jerr;

      if(jacobians[refIdx]) {
        JerrRef.block<3,3>(startIdx, 0) = -T_S0W.C();
        JerrRef.block<3, 3>(startIdx, 0 + 3) = T_S0W.C()
                                               * okvis::kinematics::crossMx(T_WS.r() - T_WS0.r());
        JerrRef.block<3,3>(startIdx+3, 0+3) = -Jerr.bottomRightCorner<3,3>();
      }

      if(jacobiansMinimal){
        if(jacobiansMinimal[idx]) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> JminMapped(
                jacobiansMinimal[idx]);
          JminMapped = Jmin;
        }
      }

      if(jacobians[idx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[idx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[idx]);
        J = Jmin * J_lift;
      }
    }

    if(jacobians[refIdx]) {

      const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jmin =  J_ * JerrRef;

      if(jacobiansMinimal) {
        if(jacobiansMinimal[refIdx]) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> JminMapped(
                jacobiansMinimal[refIdx]);
          JminMapped = Jmin;
        }
      }

      if(jacobians[refIdx]) {

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[refIdx], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[refIdx]);
        J = Jmin * J_lift;
      }
    }
  }

  return true;
}

} 
} 
