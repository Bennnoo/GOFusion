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
 * @file ViGraph.cpp
 * @brief Source file for the ViGraph class.
 * @author Stefan Leutenegger
 */

#include <okvis/ViGraph.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include "ceres/covariance.h"
#include <cmath>
#include <chrono>
#include <Eigen/SVD>
#include <random>
#include <ceres/autodiff_cost_function.h>

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

namespace okvis {

struct FourDoFResidual {
  FourDoFResidual(const Eigen::Vector3d& point_G, const Eigen::Vector3d& point_W)
      : pG_(point_G), pW_(point_W) {}

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {

    const Eigen::Matrix<T,3,1> r_GW(params[0], params[1], params[2]);
    const Eigen::Quaternion<T> q_GW(params[6], params[3],params[4], params[5]);

    Eigen::Matrix<T,3,1> pW_in_G = q_GW * pW_.template cast<T>() + r_GW;
    Eigen::Matrix<T,3,1> error = pG_.template cast<T>() - pW_in_G;

    residuals[0] = error[0];
    residuals[1] = error[1];
    residuals[2] = error[2];
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  const Eigen::Vector3d pG_;
  const Eigen::Vector3d pW_;
};

void Align4DoF_Ceres(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points_G,
                     const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points_W,
                     kinematics::Transformation& T_GW_init,
                     kinematics::Transformation& T_GW_refined)
{
  if (points_G.size() != points_W.size() || points_G.empty()) {
    std::cerr << "Point clouds must match in size and not be empty.\n";
    return;
  }

  ::ceres::Problem problem;

  okvis::ceres::PoseParameterBlock gpsParameterBlock(T_GW_init,0,okvis::Time(0));
  auto* gpsExtrinsicLocalParametrisation = new okvis::ceres::PoseManifold4d();
  gpsParameterBlock.setLocalParameterizationPtr(gpsExtrinsicLocalParametrisation);
  problem.AddParameterBlock(gpsParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension, gpsExtrinsicLocalParametrisation);

  for (size_t i = 0; i < points_G.size(); ++i) {
    ::ceres::CostFunction* cost_fn =
      new ::ceres::AutoDiffCostFunction<FourDoFResidual, 3, 7>(
        new FourDoFResidual(points_G[i], points_W[i]));
    ::ceres::LossFunction* loss_fn = new ::ceres::CauchyLoss(3.0);
    problem.AddResidualBlock(cost_fn, loss_fn, gpsParameterBlock.parameters());
  }

  ::ceres::Solver::Options options;
  options.linear_solver_type = ::ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 100;

  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  T_GW_refined = gpsParameterBlock.estimate();
}

kinematics::Transformation umeyamaTransform(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& gpsPoints,
                                            const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& worldPoints)
{

  if(gpsPoints.size() < 3 || (gpsPoints.size() != worldPoints.size())){
    LOG(ERROR) << "Umeyama cannot be computed!";
    return kinematics::Transformation::Identity();
  }

  Eigen::Vector3d centroidGps, centroidWorld;
  Eigen::MatrixXd gpsPtMatrix(3, gpsPoints.size());
  Eigen::MatrixXd worldPtMatrix(3, worldPoints.size());
  centroidGps.setZero();
  centroidWorld.setZero();
  gpsPtMatrix.setZero();
  worldPtMatrix.setZero();

  for(size_t i = 0; i < gpsPoints.size() ; ++i){
    gpsPtMatrix.col(i) = gpsPoints.at(i);
    worldPtMatrix.col(i) = worldPoints.at(i);

    centroidGps += gpsPoints.at(i);
    centroidWorld += worldPoints.at(i);

  }
  centroidGps /= gpsPoints.size();
  centroidWorld /= worldPoints.size();

  gpsPtMatrix.colwise() -= centroidGps;
  worldPtMatrix.colwise() -= centroidWorld;

  Eigen::Matrix3d H;
  H = worldPtMatrix * gpsPtMatrix.transpose();

  double A = H(0,1) - H(1,0);
  double B = H(0,0) + H(1,1);
  double theta = M_PI / 2.0 - std::atan2(B,A);
  Eigen::Matrix3d R_yaw;
  R_yaw.setZero();
  R_yaw(0,0)=std::cos(theta);
  R_yaw(0,1) = - std::sin(theta);
  R_yaw(1,0) = std::sin(theta);
  R_yaw(1,1) = std::cos(theta);
  R_yaw(2,2) = 1.0;

  Eigen::Vector3d t = centroidGps - R_yaw * centroidWorld;

  Eigen::Matrix4d T;
  T.setIdentity();
  T.topLeftCorner<3,3>() = R_yaw;
  T.topRightCorner<3,1>() = t;

  return kinematics::Transformation(T);
}

struct RigidResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double inlier_ratio;
  std::vector<int> inliers;
};

RigidResult estimateRigidRansac(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& gpsPoints,
                                const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& worldPoints,
                                int iterations = 10,
                                size_t n_points = 6,
                                double inlierThreshold = 0.5,
                                double requiredInlierRatio = 0.7)
{
  if(gpsPoints.size() < 2*n_points) return RigidResult();
  assert(gpsPoints.size() == worldPoints.size());
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, gpsPoints.size() - 1);

  RigidResult best;
  best.inlier_ratio = 0.0;

  for (int it = 0; it < iterations; ++it) {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> gpsSubset, worldSubset;
    std::vector<int> idxs;
    while (idxs.size() < n_points) {
      int idx = dist(rng);
      if (std::find(idxs.begin(), idxs.end(), idx) == idxs.end())
        idxs.push_back(idx);
    }
    for (int i : idxs) {
      gpsSubset.push_back(gpsPoints[i]);
      worldSubset.push_back(worldPoints[i]);
    }

    kinematics::Transformation T_align = umeyamaTransform(gpsSubset, worldSubset);
    Eigen::Vector3d t = T_align.r();
    Eigen::Matrix3d R_yaw = T_align.C();

    std::vector<int> inliers;
    for (size_t i = 0; i < gpsPoints.size(); ++i) {
      Eigen::Vector3d est = R_yaw * worldPoints[i] + t;
      double err = (gpsPoints[i] - est).norm();
      if (err < inlierThreshold)
        inliers.push_back(i);
    }

    double ratio = static_cast<double>(inliers.size()) / gpsPoints.size();

    if (ratio > best.inlier_ratio) {
      best.inlier_ratio = ratio;
      best.R = R_yaw;
      best.t = t;
      best.inliers = inliers;
    }

    if (ratio > requiredInlierRatio)
      break;
  }

  return best;
}

ViGraph::ViGraph() : globCartesianFrame_(earth_)
{
  cauchyLossFunctionPtr_.reset(new ::ceres::CauchyLoss(1.0));
  cauchyGpsLossFunctionPtr_.reset(new ::ceres::CauchyLoss(3.0));
  tukeyDepthLossFunctionPtr_.reset(new ::ceres::TukeyLoss(0.1));
  tukeyLidarLossFunctionPtr_.reset(new ::ceres::TukeyLoss(2.0));
  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.enable_fast_removal = true;
  problem_.reset(new ::ceres::Problem(problemOptions));
  options_.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
  options_.trust_region_strategy_type = ::ceres::DOGLEG;

}

int ViGraph::addCamera(const CameraParameters& cameraParameters) {
  cameraParametersVec_.push_back(cameraParameters);
  return static_cast<int>(cameraParametersVec_.size()) - 1;
}

int ViGraph::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return static_cast<int>(imuParametersVec_.size()) - 1;
}

int ViGraph::addGps(const GpsParameters& gpsParameters) {
  if (gpsParametersVec_.size() > 1) {
    LOG(ERROR) << "only one GPS currently supported";
    return -1;
  }
  gpsParametersVec_.push_back(gpsParameters);
  return static_cast<int>(gpsParametersVec_.size()) - 1;
}

StateId ViGraph::addStatesInitialise(
    const Time &timestamp, const ImuMeasurementDeque &imuMeasurements,
    const cameras::NCameraSystem & nCameraSystem)
{
  State state;
  state.timestamp = timestamp;
  state.isKeyframe = true; 
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.size()==0, "states added before...")
  StateId id(1);

  kinematics::Transformation T_WS;
  T_WS.setIdentity();

  if (imuParametersVec_.at(0).use) {
    OKVIS_ASSERT_TRUE_DBG(Exception, imuMeasurements.size() > 0, "no IMU measurements passed")

    Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
    for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
         it < imuMeasurements.end();
         ++it) {
      acc_B += it->measurement.accelerometers;
    }
    acc_B /= double(imuMeasurements.size());
    Eigen::Vector3d e_acc = acc_B.normalized();

    Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
    Eigen::Matrix<double, 6, 1> poseIncrement;
    poseIncrement.head<3>() = Eigen::Vector3d::Zero();
    poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
    double angle = std::acos(ez_W.transpose() * e_acc);
    poseIncrement.tail<3>() *= angle;
    T_WS.oplus(-poseIncrement);
  } else {

    kinematics::Transformation T_WC;
    T_WC.set(T_WC.r(), T_WC.q() * Eigen::Quaterniond(-sqrt(2), sqrt(2), 0, 0));
    T_WS = T_WC * nCameraSystem.T_SC(0)->inverse();
  }

  state.pose.reset(new ceres::PoseParameterBlock(T_WS, id.value(), timestamp));
  SpeedAndBias speedAndBias = SpeedAndBias::Zero();
  speedAndBias.tail<3>() = imuParametersVec_.at(0).a0;
  speedAndBias.segment<3>(3) = imuParametersVec_.at(0).g0;
  state.speedAndBias.reset(
        new ceres::SpeedAndBiasParameterBlock(speedAndBias, id.value(), timestamp));
  problem_->AddParameterBlock(state.pose->parameters(), 7, &poseManifold_);
  state.pose->setLocalParameterizationPtr(&poseManifold_);
  problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);
  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {
    const kinematics::Transformation T_SC = *nCameraSystem.T_SC(i);
    state.extrinsics.push_back(std::shared_ptr<ceres::PoseParameterBlock>(
                                 new ceres::PoseParameterBlock(T_SC, id.value(), timestamp)));
    problem_->AddParameterBlock(state.extrinsics.back()->parameters(), 7,
                                &poseManifold_);
    state.extrinsics.back()->setLocalParameterizationPtr(&poseManifold_);
  }

  kinematics::Transformation T_GW0;
  T_GW0.setIdentity();
  state.T_GW.reset(new ceres::PoseParameterBlock(T_GW0,id.value(),timestamp));
  problem_->AddParameterBlock(state.T_GW->parameters(),7,&gpsExtrinsicLocalParametrisation_);
  state.T_GW->setLocalParameterizationPtr(&gpsExtrinsicLocalParametrisation_);
  problem_->SetParameterBlockVariable(state.T_GW->parameters());

  Eigen::Matrix<double, 6, 1> informationDiag = Eigen::Matrix<double, 6, 1>::Ones();
  informationDiag[0] = 1.0e8;
  informationDiag[1] = 1.0e8;
  informationDiag[2] = 1.0e8;
  if(imuParametersVec_.at(0).use) {

    informationDiag[3] = 0.0;
    informationDiag[4] = 0.0;
  } else {

    informationDiag[3] = 1.0e2;
    informationDiag[4] = 1.0e2;
  }
  informationDiag[5] = 1.0e2;
  state.posePrior.errorTerm.reset(new ceres::PoseError(T_WS, informationDiag));
  const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
  const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
  state.speedAndBiasPrior.errorTerm.reset(
        new ceres::SpeedAndBiasError(speedAndBias, 0.1, sigma_bg * sigma_bg, sigma_ba * sigma_ba));
  state.posePrior.residualBlockId = problem_->AddResidualBlock(
        state.posePrior.errorTerm.get(), nullptr, state.pose->parameters());
  state.speedAndBiasPrior.residualBlockId = problem_->AddResidualBlock(
        state.speedAndBiasPrior.errorTerm.get(), nullptr, state.speedAndBias->parameters());
  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {
    if(cameraParametersVec_.at(i).online_calibration.do_extrinsics) {

      PosePrior extrinsicsPrior;
      const double sigma_r = cameraParametersVec_.at(i).online_calibration.sigma_r;
      const double sigma_alpha = cameraParametersVec_.at(i).online_calibration.sigma_alpha;
      extrinsicsPrior.errorTerm.reset(
            new ceres::PoseError(
              state.extrinsics.at(i)->estimate(), sigma_r*sigma_r, sigma_alpha*sigma_alpha));
      extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
                extrinsicsPrior.errorTerm.get(), nullptr, state.extrinsics.at(i)->parameters());
      state.extrinsicsPriors.push_back(extrinsicsPrior);
    } else {

      problem_->SetParameterBlockConstant(state.extrinsics.at(i)->parameters());
      state.extrinsics.at(i)->setFixed(true);
    }
  }

  states_[id] = state; 
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[id] = anyState; 

  return id;
}

StateId ViGraph::addStatesPropagate(const Time &timestamp,
                                     const ImuMeasurementDeque &imuMeasurements, bool isKeyframe)
{

  State state;
  state.timestamp = timestamp;
  state.isKeyframe = isKeyframe;
  StateId id(states_.rbegin()->first+1);
  State & lastState = states_.rbegin()->second;
  kinematics::Transformation T_WS = lastState.pose->estimate();
  SpeedAndBias speedAndBias = lastState.speedAndBias->estimate();
  if(imuParametersVec_.at(0).use) {
    ceres::ImuError::propagation(imuMeasurements,
                                         imuParametersVec_.at(0),
                                         T_WS,
                                         speedAndBias,
                                         lastState.timestamp,
                                         timestamp);
  } else {

    auto iter = states_.rbegin();
    ++iter;
    if (iter != states_.rend()) {
      const double r = (timestamp - lastState.timestamp).toSec()
                       / (lastState.timestamp - iter->second.timestamp).toSec();
      kinematics::Transformation T_WS_m1 = iter->second.pose->estimate();
      Eigen::Vector3d dr = r * (T_WS.r() - T_WS_m1.r());
      Eigen::AngleAxisd daa(T_WS.q() * T_WS_m1.q().inverse());
      daa.angle() *= r;
      T_WS.set(T_WS.r() + dr, T_WS.q() * Eigen::Quaterniond(daa));
    }
  }
  state.pose.reset(new ceres::PoseParameterBlock(T_WS, id.value(), timestamp));
  problem_->AddParameterBlock(state.pose->parameters(), 7, &poseManifold_);
  state.pose->setLocalParameterizationPtr(&poseManifold_);
  state.speedAndBias.reset(new ceres::SpeedAndBiasParameterBlock(
                             speedAndBias, id.value(), timestamp));
  problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);

  ImuLink imuLink;
  if(imuParametersVec_.at(0).use) {
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          lastState.timestamp > imuMeasurements.front().timeStamp,
                          "Cannot propagate IMU: " << lastState.timestamp << ">"
                                                   << imuMeasurements.front().timeStamp)
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          timestamp < imuMeasurements.back().timeStamp,
                          "Cannot propagate IMU: " << timestamp << "<"
                                                   << imuMeasurements.back().timeStamp)
    imuLink.errorTerm.reset(new ceres::ImuError(imuMeasurements, imuParametersVec_.at(0),
                                                lastState.timestamp, timestamp));
  } else {
    imuLink.errorTerm.reset(new ceres::PseudoImuError(lastState.timestamp, timestamp));
  }

  imuLink.residualBlockId = problem_->AddResidualBlock(
    imuLink.errorTerm.get(), nullptr,
    lastState.pose->parameters(), lastState.speedAndBias->parameters(),
    state.pose->parameters(), state.speedAndBias->parameters());

  lastState.nextImuLink = imuLink;
  state.previousImuLink = imuLink;

  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {

    state.extrinsics.push_back(lastState.extrinsics.at(i));
  }

  state.T_GW = lastState.T_GW;
  state.GpsFactors.clear();

  states_[id] = state; 
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[id] = anyState; 

  return id;
}

bool ViGraph::addStatesFromOther(StateId stateId, const ViGraph &other)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, other.states_.count(stateId), "stateId not found")

  State state;
  const State& otherState = other.states_.at(stateId);
  state.timestamp = otherState.timestamp;
  state.isKeyframe = otherState.isKeyframe;

  const size_t numCameras = otherState.extrinsics.size();
  state.extrinsics.resize(numCameras);

  state.pose.reset(new ceres::PoseParameterBlock(
                     otherState.pose->estimate(), stateId.value(), otherState.timestamp));
  state.speedAndBias.reset(new ceres::SpeedAndBiasParameterBlock(
                     otherState.speedAndBias->estimate(), stateId.value(), otherState.timestamp));
  for(size_t i = 0; i < numCameras; ++i) {
    state.extrinsics.at(i).reset(
            new ceres::PoseParameterBlock(otherState.extrinsics.at(i)->estimate(),
                                          stateId.value(), otherState.timestamp));
  }

  state.T_GW = otherState.T_GW;

  if(otherState.posePrior.errorTerm) {
    PosePrior& posePrior = state.posePrior;
    posePrior.errorTerm.reset(new ceres::PoseError(otherState.posePrior.errorTerm->measurement(),
                                                   otherState.posePrior.errorTerm->information()));
    posePrior.residualBlockId = problem_->AddResidualBlock(posePrior.errorTerm.get(),
                                                           nullptr, state.pose->parameters());
  }
  if(otherState.speedAndBiasPrior.errorTerm) {
    SpeedAndBiasPrior& speedAndBiasPrior = state.speedAndBiasPrior;
    speedAndBiasPrior.errorTerm.reset(
          new ceres::SpeedAndBiasError(otherState.speedAndBiasPrior.errorTerm->measurement(),
                                       otherState.speedAndBiasPrior.errorTerm->information()));
    speedAndBiasPrior.residualBlockId = problem_->AddResidualBlock(
          speedAndBiasPrior.errorTerm.get(), nullptr, state.speedAndBias->parameters());
  }
  for(size_t i=0; i<otherState.extrinsicsPriors.size(); ++i) {
    PosePrior extrinsicsPrior;
    const PosePrior& otherExtrinsicsPrior = state.extrinsicsPriors.at(i);
    extrinsicsPrior.errorTerm.reset(
          new ceres::PoseError(otherExtrinsicsPrior.errorTerm->measurement(),
                               otherExtrinsicsPrior.errorTerm->information()));
    extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
          extrinsicsPrior.errorTerm.get(), nullptr, state.extrinsics.at(i)->parameters());
    state.extrinsicsPriors.push_back(extrinsicsPrior);
  }

  if(otherState.previousImuLink.errorTerm) {
    auto otherIter = other.states_.find(stateId);
    OKVIS_ASSERT_TRUE_DBG(Exception, otherIter != other.states_.begin(), "no previous state")
    otherIter--;
    const State& otherPreviousState = otherIter->second;
    const Time t_0 = otherPreviousState.timestamp;
    const Time t_1 = otherState.timestamp;
    State& previousState = states_.rbegin()->second;
    OKVIS_ASSERT_TRUE_DBG(Exception, otherIter->first == states_.rbegin()->first,
                      "different previous states")
    OKVIS_ASSERT_TRUE_DBG(Exception, t_0 == previousState.timestamp, "inconsistent previous times")
    ImuLink imuLink;
    imuLink.errorTerm = otherState.previousImuLink.errorTerm->clone();
    imuLink.residualBlockId = problem_->AddResidualBlock(
          imuLink.errorTerm.get(), nullptr,
          previousState.pose->parameters(), previousState.speedAndBias->parameters(),
          state.pose->parameters(), state.speedAndBias->parameters());
    state.previousImuLink = imuLink;
    previousState.nextImuLink = imuLink;

    for(size_t i=0; i<otherState.previousExtrinsicsLink.size(); ++i) {
      ExtrinsicsLink extrinsicsLink;
      const ceres::RelativePoseError& otherRelativePoseError =
          *otherState.previousExtrinsicsLink.at(i).errorTerm;
      extrinsicsLink.errorTerm.reset(
            new ceres::RelativePoseError(otherRelativePoseError.information()));
      extrinsicsLink.residualBlockId = problem_->AddResidualBlock(
            extrinsicsLink.errorTerm.get(), nullptr, previousState.extrinsics.at(i)->parameters(),
            state.extrinsics.at(i)->parameters());
      state.previousExtrinsicsLink.push_back(extrinsicsLink);
      previousState.nextExtrinsicsLink.push_back(extrinsicsLink);
    }
  }

  states_[stateId] = state; 
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[stateId] = anyState; 

  return true;
}

bool ViGraph::addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &homogeneousPoint,
                           bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarkId.isInitialised(), "landmark ID invalid")
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId) == 0, "landmark already exists")
  Landmark landmark;
  landmark.hPoint.reset(new ceres::HomogeneousPointParameterBlock(
                          homogeneousPoint, landmarkId.value(), initialised));
  problem_->AddParameterBlock(landmark.hPoint->parameters(), 4,
                              &homogeneousPointManifold_);
  landmark.hPoint->setLocalParameterizationPtr(&homogeneousPointManifold_);
  landmarks_[landmarkId] = landmark;
  return true;
}

LandmarkId ViGraph::addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  const LandmarkId landmarkId = landmarks_.empty() ?
        LandmarkId(1) : LandmarkId(landmarks_.rbegin()->first+1); 
  Landmark landmark;
  landmark.hPoint.reset(new ceres::HomogeneousPointParameterBlock(
                          homogeneousPoint, landmarkId.value(), initialised));
  problem_->AddParameterBlock(landmark.hPoint->parameters(), 4,
                              &homogeneousPointManifold_);
  landmark.hPoint->setLocalParameterizationPtr(&homogeneousPointManifold_);
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId) == 0, "Bug: landmark not added")
  landmarks_[landmarkId] = landmark;
  return landmarkId;
}

bool ViGraph::removeLandmark(LandmarkId landmarkId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  Landmark& landmark = landmarks_.at(landmarkId);

  for(auto observation : landmark.observations) {
    problem_->RemoveResidualBlock(observation.second.residualBlockId);
    if (observation.second.depthError.residualBlockId) {
      problem_->RemoveResidualBlock(observation.second.depthError.residualBlockId);
    }
    observations_.erase(observation.first);

    StateId stateId(observation.first.frameId);
    states_.at(stateId).observations.erase(observation.first);
  }

  problem_->RemoveParameterBlock(landmark.hPoint->parameters());
  landmarks_.erase(landmarkId);
  return true;
}

bool ViGraph::setLandmarkInitialised(LandmarkId landmarkId, bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  landmarks_.at(landmarkId).hPoint->setInitialized(initialised);
  return true;
}

bool ViGraph::isLandmarkInitialised(LandmarkId landmarkId) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  return landmarks_.at(landmarkId).hPoint->initialized();
}

bool ViGraph::isLandmarkAdded(LandmarkId landmarkId) const
{
  return landmarks_.count(landmarkId)>0;
}

void ViGraph::checkObservations() const {

  for(auto obs : observations_) {
    OKVIS_ASSERT_TRUE(Exception,
                      landmarks_.at(obs.second.landmarkId).observations.count(obs.first),
                      "observations check failed")
    OKVIS_ASSERT_TRUE(Exception,
                      states_.at(StateId(obs.first.frameId)).observations.count(obs.first),
                      "observations check failed")
  }

  for(auto state : states_) {
    for(auto obs : state.second.observations) {
      OKVIS_ASSERT_TRUE(Exception,
                        landmarks_.at(obs.second.landmarkId).observations.count(obs.first),
                        "observations check failed")
      OKVIS_ASSERT_TRUE(Exception, observations_.count(obs.first), "observations check failed")
    }
  }

  for(auto lm : landmarks_) {
    for(auto obs : lm.second.observations) {
      OKVIS_ASSERT_TRUE(Exception, observations_.count(obs.first), "observations check failed")
      OKVIS_ASSERT_TRUE(Exception,
                        states_.at(StateId(obs.first.frameId)).observations.count(obs.first),
                        "observations check failed");
    }
  }

}

bool ViGraph::removeObservation(KeypointIdentifier keypointId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, observations_.count(keypointId), "observation does not exists")
  Observation observation = observations_.at(keypointId);

  problem_->RemoveResidualBlock(observation.residualBlockId);
  if (observation.depthError.residualBlockId) {
    problem_->RemoveResidualBlock(observation.depthError.residualBlockId);
  }

  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(observation.landmarkId),
                        "landmark does not exists")
  Landmark& landmark = landmarks_.at(observation.landmarkId);
  OKVIS_ASSERT_TRUE_DBG(Exception, landmark.observations.count(keypointId),
                    "observation does not exists")
  landmark.observations.erase(keypointId);
  StateId stateId(keypointId.frameId);
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId),
                    "state does not exists")
  State& state = states_.at(stateId);
  OKVIS_ASSERT_TRUE_DBG(Exception, state.observations.count(keypointId),
                    "observation does not exists")
  state.observations.erase(keypointId);
  observations_.erase(keypointId);

  covisibilitiesComputed_ = false;

  return true;
}

bool ViGraph::computeCovisibilities()
{
  if(covisibilitiesComputed_) {
    return true; 
  }
  coObservationCounts_.clear();
  visibleFrames_.clear();
  for(auto iter=landmarks_.begin(); iter!=landmarks_.end(); ++iter) {

    auto obs = iter->second.observations;
    std::set<uint64> covisibilities;
    for(auto obsiter=obs.begin(); obsiter!=obs.end(); ++obsiter) {
      covisibilities.insert(obsiter->first.frameId);
      visibleFrames_.insert(StateId(obsiter->first.frameId));
    }
    for(auto i0=covisibilities.begin(); i0!=covisibilities.end(); ++i0) {
      for(auto i1=covisibilities.begin(); i1!=covisibilities.end(); ++i1) {
        if(*i1>=*i0) {
          continue;
        }
        if(coObservationCounts_.find(*i0)==coObservationCounts_.end()) {
          coObservationCounts_[*i0][*i1] = 1;
        } else {
          if (coObservationCounts_.at(*i0).find(*i1)==coObservationCounts_.at(*i0).end()) {
            coObservationCounts_.at(*i0)[*i1] = 1;
          } else {
            coObservationCounts_.at(*i0).at(*i1)++;
          }
        }
      }
    }
  }
  covisibilitiesComputed_ = true;
  return coObservationCounts_.size()>0;
}

int ViGraph::covisibilities(StateId pose_i, StateId pose_j) const
{
  OKVIS_ASSERT_TRUE(Exception, covisibilitiesComputed_, "covisibilities not yet computed")
  if(pose_i==pose_j) {
    return 0;
  }
  size_t a=pose_i.value();
  size_t b=pose_j.value();
  if(pose_i < pose_j) {
    b=pose_i.value();
    a=pose_j.value();
  }
  if(coObservationCounts_.count(a)) {
    if(coObservationCounts_.at(a).count(b)) {
      return coObservationCounts_.at(a).at(b);
    }
  }
  return 0;
}

bool ViGraph::addRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                         const kinematics::Transformation &T_S0S1,
                                         const Eigen::Matrix<double, 6, 6> &information)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId0), "stateId not found")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId1), "stateId not found")
  State & state0 = states_.at(poseId0);
  State & state1 = states_.at(poseId1);
  OKVIS_ASSERT_TRUE_DBG(Exception, state0.relativePoseLinks.count(poseId1)==0,
                    "relative pose error already exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, state1.relativePoseLinks.count(poseId0)==0,
                    "relative pose error already exists")
  RelativePoseLink relativePoseLink;
  relativePoseLink.errorTerm.reset(new ceres::RelativePoseError(information, T_S0S1));
  relativePoseLink.residualBlockId = problem_->AddResidualBlock(relativePoseLink.errorTerm.get(),
                                                                nullptr, state0.pose->parameters(),
                                                                state1.pose->parameters());
  relativePoseLink.state0 = poseId0;
  relativePoseLink.state1 = poseId1;

  state0.relativePoseLinks[poseId1] = relativePoseLink;
  state1.relativePoseLinks[poseId0] = relativePoseLink;

  return true;
}

bool ViGraph::removeRelativePoseConstraint(StateId poseId0, StateId poseId1)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId0),
                        "stateId " << poseId0.value() << " not found")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId1),
                        "stateId " << poseId1.value() << "not found")
  State & state0 = states_.at(poseId0);
  State & state1 = states_.at(poseId1);
  OKVIS_ASSERT_TRUE_DBG(Exception, state0.relativePoseLinks.count(poseId1)==1,
                    "relative pose error does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, state1.relativePoseLinks.count(poseId0)==1,
                    "relative pose error does not exists")
  RelativePoseLink & relativePoseLink = state0.relativePoseLinks.at(poseId1);
  problem_->RemoveResidualBlock(relativePoseLink.residualBlockId);

  state0.relativePoseLinks.erase(poseId1);
  state1.relativePoseLinks.erase(poseId0);

  return true;
}

void ViGraph::freezeGpsExtrinsics(){
    problem_->SetParameterBlockConstant(states_.begin()->second.T_GW->parameters());
    gpsFixed_ = true;
    LOG(INFO) << "[GPS] Freezing GPS Extrinsics!";
}

void ViGraph::unfreezeGpsExtrinsics(){
    problem_->SetParameterBlockVariable(states_.begin()->second.T_GW->parameters());
    gpsFixed_ = false;
    LOG(INFO) << "[GPS] Unfreezing GPS Extrinsics!";
}

bool ViGraph::setGpsExtrinsics(const kinematics::TransformationCacheless & T_GW){
    states_.begin()->second.T_GW->setEstimate(T_GW);
    return true;
}

bool ViGraph::needsGpsReInit(){
    if(gpsStates_.empty())
        return false;

    StateId lastGpsStateId;
    lastGpsStateId = *(gpsStates_.rbegin());

    if(gpsStatus_ == gpsStatus::Initialised && states_.at(lastGpsStateId).pose->fixed()){
        needsPositionAlignment_ = true;
        gpsDropoutId_ = lastGpsStateId;
        return true;
      }
    if(gpsStatus_ == gpsStatus::ReInitialising  && states_.lower_bound(positionAlignedId_)->second.pose->fixed()){
        needsPositionAlignment_ = true;
        return true;
      }

    return false;
}

void ViGraph::reInitGpsExtrinsics(){

    gpsStatus_ = gpsStatus::ReInitialising;
    needsPositionAlignment_ = true;
}

bool ViGraph::needsFullGpsAlignment(StateId& gpsLossId, StateId& gpsAlignId, okvis::kinematics::Transformation& T_GW_new){
    if(needsFullAlignment_){
      gpsLossId = gpsDropoutId_;
      gpsAlignId = *gpsReInitStates_.rbegin();
      T_GW_new = T_GW_init_;

      return true;
    }
    else{
      return false;
    }
}

bool ViGraph::needsPosGpsAlignment(StateId& gpsLossId, StateId& gpsAlignId, Eigen::Vector3d& posError){

    if(gpsParametersVec_.back().robustGpsInit){
        return false;
    }

    if(needsPositionAlignment_){

        gpsLossId = gpsDropoutId_;

        GpsMeasurement lastMeasurement = gpsInitMap_.rbegin()->second;
        if(gpsParametersVec_.back().type == "geodetic" || gpsParametersVec_.back().type == "geodetic-leica") {

          double x,y,z;
          globCartesianFrame_.Forward(lastMeasurement.measurement.latitude, lastMeasurement.measurement.longitdue, lastMeasurement.measurement.height,
                                      x, y, z);
          lastMeasurement.measurement.setPosition(x, y, z);
        }
        auto rIter = states_.rbegin();

        while((lastMeasurement.timeStamp < rIter->second.timestamp) && (rIter != states_.rend()))
          ++rIter;

        if(rIter == states_.rend())
          return false;

        okvis::ceres::GpsErrorAsynchronous test(lastMeasurement.measurement.position, lastMeasurement.measurement.covariances.inverse(),
                                                gpsInitImuQueue_, imuParametersVec_.back(), rIter->second.timestamp, lastMeasurement.timeStamp,
                                                gpsParametersVec_.back());
        double* parameters[3];
        parameters[0]=states_.at(rIter->first).pose->parameters();
        parameters[1]=states_.at(rIter->first).speedAndBias->parameters();
        parameters[2]=states_.begin()->second.T_GW->parameters();
        Eigen::Matrix<double,3,1> residuals;
        okvis::kinematics::Transformation T_GW_original = states_.begin()->second.T_GW->estimate();

        test.EvaluateWithMinimalJacobians(parameters, residuals.data(),NULL,NULL);
        Eigen::Vector3d posError_G = test.error();

        posError = T_GW_original.inverse().C() * posError_G;

        gpsAlignId = rIter->first;
        positionAlignedId_ = rIter->first;

        return true;

      }
    else{
        return false;
      }
}

bool ViGraph::needsInitialGpsAlignment(){
  return needsInitialAlignment_;
}

void ViGraph::resetFullGpsAlignment(){

    gpsReInitialised_ = false;
    needsFullAlignment_ = false;
    needsPositionAlignment_ = false; 
    gpsStatus_ = gpsStatus::Initialised;

    gpsInitMap_.clear();
    gpsInitImuQueue_.clear();
    gpsReInitStates_.clear();
}

void ViGraph::resetPosGpsAlignment(){
    needsPositionAlignment_=false;
}

bool ViGraph::addGpsMeasurement(StateId poseId, GpsMeasurement &gpsMeas, const ImuMeasurementDeque &imuMeasurements){

    OKVIS_ASSERT_TRUE(Exception, states_.count(poseId), "stateId " << poseId.value() << " not found")
    OKVIS_ASSERT_TRUE(Exception, (gpsMeas.timeStamp >= states_.at(poseId).timestamp), "GPS measurement too old to add to state" )
    if(!(imuMeasurements.front().timeStamp <= states_.at(poseId).timestamp)){
      LOG(WARNING) << "IMU Measurements for adding GPS error are not old enough" << std::endl;
      return false;
    }

    OKVIS_ASSERT_TRUE(Exception, (imuMeasurements.back().timeStamp >= gpsMeas.timeStamp), "IMU measurements do not cover GPS measurement");

    gpsStates_.insert(poseId); 
    State & state = states_.at(poseId); 
    GpsFactor newGpsFactor; 

    if(gpsParametersVec_.back().type == "geodetic" || gpsParametersVec_.back().type == "geodetic-leica"){
      double x,y,z;
      globCartesianFrame_.Forward(gpsMeas.measurement.latitude, gpsMeas.measurement.longitdue, gpsMeas.measurement.height,
                                  x, y, z);
      gpsMeas.measurement.setPosition(x, y, z);
    }

    newGpsFactor.errorTerm.reset(new ceres::GpsErrorAsynchronous(gpsMeas.measurement.position, gpsMeas.measurement.covariances.inverse(),
                                     imuMeasurements, imuParametersVec_.back(),state.timestamp, gpsMeas.timeStamp, gpsParametersVec_.back()));
    if(gpsStatus_ == gpsStatus::Initialising || gpsStatus_ == gpsStatus::Initialised || gpsStatus_ == gpsStatus::ReInitialising) {
      newGpsFactor.residualBlockId = problem_->AddResidualBlock(newGpsFactor.errorTerm.get(), cauchyGpsLossFunctionPtr_.get(),
                                                                   state.pose->parameters(),state.speedAndBias->parameters(), state.T_GW->parameters());
    }
    state.GpsFactors.push_back(newGpsFactor);

    return true;

}

void ViGraph::gpsMeasurements(StateId stateId, AlignedVector<Eigen::Vector3d>& gpsMeasurements){
  for(auto iter = states_.at(stateId).GpsFactors.begin(); iter != states_.at(stateId).GpsFactors.end(); ++iter){
    gpsMeasurements.push_back(iter->errorTerm->measurement());
  }
}

bool ViGraph::checkForGpsInit(okvis::kinematics::Transformation& T_GW, std::set<StateId> consideredStates, double* yaw_error) {

  if(consideredStates.size() < 2)
    return false;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > gpsPoints;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> worldPoints;
  std::vector<Eigen::Matrix<double,3,3>, Eigen::aligned_allocator<Eigen::Matrix<double,3,3>> > covariances;

  auto gpsStateIter = consideredStates.begin();
  if(consideredStates.size() > 100 && gpsParametersVec_.back().robustGpsInit){
    gpsStateIter = std::prev(consideredStates.end(), 100);
  }

  for(; gpsStateIter!= consideredStates.end(); gpsStateIter++){
    auto gpsState = *gpsStateIter;

    okvis::kinematics::Transformation T_WS_state = states_.at(gpsState).pose->estimate();
    okvis::SpeedAndBias sb_state = states_.at(gpsState).speedAndBias->estimate();

    for(auto iter = states_.at(gpsState).GpsFactors.begin(); iter != states_.at(gpsState).GpsFactors.end(); ++iter){
      gpsPoints.push_back(iter->errorTerm->measurement());
      okvis::kinematics::Transformation T_WS_prop;
      iter->errorTerm->applyPreInt(T_WS_state,sb_state,T_WS_prop);
      worldPoints.push_back(T_WS_prop.r() + T_WS_prop.C() * gpsParametersVec_.back().r_SA);
      covariances.push_back(iter->errorTerm->covariance());
    }
  }

  Eigen::Vector3d centroidGps, centroidWorld;
  Eigen::MatrixXd gpsPtMatrix(3, gpsPoints.size());
  Eigen::MatrixXd worldPtMatrix(3, worldPoints.size());
  centroidGps.setZero();
  centroidWorld.setZero();
  gpsPtMatrix.setZero();
  worldPtMatrix.setZero();

  for(size_t i = 0; i < gpsPoints.size() ; ++i){
    gpsPtMatrix.col(i) = gpsPoints.at(i);
    worldPtMatrix.col(i) = worldPoints.at(i);

    centroidGps += gpsPoints.at(i);
    centroidWorld += worldPoints.at(i);

  }
  centroidGps /= gpsPoints.size();
  centroidWorld /= worldPoints.size();

  gpsPtMatrix.colwise() -= centroidGps;
  worldPtMatrix.colwise() -= centroidWorld;

  if(gpsParametersVec_.back().robustGpsInit){

    RigidResult ransac_init_result = estimateRigidRansac(gpsPoints, worldPoints, 20, 20, 4.0, 0.7);
    if(ransac_init_result.inlier_ratio < 0.25){
      DLOG(WARNING) << "Rejecting due to ransac_init_result.inlier_ratio < 0.25:  " << ransac_init_result.inlier_ratio << " / " << ransac_init_result.inliers.size()<< " / " << ransac_init_result.t.transpose()<< " / " << gpsPoints.size();
      return false;
    }
    Eigen::Matrix4d T_align;
    T_align.setIdentity();
    T_align.topLeftCorner<3,3>() = ransac_init_result.R;
    T_align.topRightCorner<3,1>() = ransac_init_result.t;
    T_GW.set(T_align);
  }
  else {
    T_GW = umeyamaTransform(gpsPoints, worldPoints);
  }

  Eigen::Matrix<double,4,4> Hess;
  Hess.setZero();

  Eigen::Matrix<double,3,4> Ei;
  Eigen::Matrix<double,4,4> tmp2;
  for(size_t i = 0; i < worldPoints.size(); ++i){

    Eigen::Vector3d pt = worldPoints.at(i);
    Ei.setZero();
    Ei.topLeftCorner<3,3>() = -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double,3,3> tmp = okvis::kinematics::crossMx(T_GW.C()*pt);
    Ei.topRightCorner<3,1>() = tmp.col(2);
    tmp2 = Ei.transpose() * covariances.at(i).inverse() * Ei;
    Hess = Hess + tmp2;
  }

  Eigen::Matrix<double,4,4> P = Hess.inverse();
  double yawUncertainty = std::sqrt(P(3,3)) / M_PI * 180.0;
  if(yaw_error) {
    *yaw_error = yawUncertainty;
  }

  if(yawUncertainty < gpsParametersVec_.back().yawErrorThreshold){

    if(gpsParametersVec_.back().robustGpsInit){

      kinematics::Transformation T_GW_refined;
      Align4DoF_Ceres(gpsPoints, worldPoints, T_GW, T_GW_refined);
      DLOG(WARNING) << "Nonlinear refinement of T_GW - starting from :\n " << T_GW.T3x4() << std::endl << " ... ending up at \n" << T_GW_refined.T3x4();
      T_GW = T_GW_refined; 
    }
    return true;
  }
  else{
    DLOG(WARNING) << "[GPS] Rejecting Initialization due to yaw error: " << yawUncertainty;
    return false;
  }
}

int ViGraph::checkValidGpsMeasurements(GpsMeasurementDeque& inputGpsMeasurementDeque, GpsMeasurementDeque& gpsMeasurementDeque){

  gpsMeasurementDeque.clear();

  if(inputGpsMeasurementDeque.size() == 0){
    return 0;
  }

  auto rIterStates = states_.rbegin();

  int countValidMeasurements = 0;
  bool needsReInit = false;
  StateId sid;
  StateId lastGpsStateId;
  if(gpsStatus_ == gpsStatus::Initialised){
    lastGpsStateId = *(gpsStates_.rbegin());
    needsReInit = states_.at(lastGpsStateId).pose->fixed();
  }
  else if (gpsStatus_ == gpsStatus::ReInitialising){
    lastGpsStateId = gpsDropoutId_;
  }

  for(auto rIterMeas = inputGpsMeasurementDeque.rbegin(); rIterMeas != inputGpsMeasurementDeque.rend() ; rIterMeas++){

      while(rIterStates->second.timestamp > rIterMeas->timeStamp && rIterStates != states_.rend()){
          rIterStates++;
      }

      if(rIterStates == states_.rend()){
          break;
      }

      sid = rIterStates->first;

      if (gpsStatus_ == gpsStatus::Off || gpsStatus_ == gpsStatus::Idle || gpsStatus_ == gpsStatus::Initialising){
        if (rIterMeas->measurement.covariances(0,0) > std::pow(6.0,2) || rIterMeas->measurement.covariances(2,2) > std::pow(10.0,2) ){
          continue;
        }
        else {
          countValidMeasurements++;
          gpsMeasurementDeque.push_back(*rIterMeas);
        }
      }

      if ((gpsStatus_ == gpsStatus::Initialised && !needsReInit)){

        okvis::kinematics::Transformation T_WS_state = states_.at(sid).pose->estimate();
        okvis::kinematics::Transformation T_GW_curr_est = states_.at(sid).T_GW->estimate();
        if(gpsParametersVec_.back().type == "geodetic" || gpsParametersVec_.back().type == "geodetic-leica"){
          double x=0,y=0,z=0;
          globCartesianFrame_.Forward(rIterMeas->measurement.latitude, rIterMeas->measurement.longitdue, rIterMeas->measurement.height,
                                      x, y, z);
          rIterMeas->measurement.setPosition(x, y, z);
        }

        Eigen::Vector3d gpsMeasInWorld = T_GW_curr_est.inverse().T3x4() * rIterMeas->measurement.position.homogeneous(); 
        Eigen::Vector3d error = (T_WS_state.r() + T_WS_state.C() * gpsParametersVec_.back().r_SA) - gpsMeasInWorld;
        double sigma_x = std::sqrt(rIterMeas->measurement.covariances(0,0));
        double sigma_y = std::sqrt(rIterMeas->measurement.covariances(1,1));
        double sigma_z = std::sqrt(rIterMeas->measurement.covariances(2,2));

        if(fabs(error.x()) > 3.0*sigma_x || fabs(error.y()) > 3.0*sigma_y || fabs(error.z()) > 3.0*sigma_z){
          continue;
        }
        else{
          countValidMeasurements++;
          gpsMeasurementDeque.push_back(*rIterMeas);
        }
      }

      else if (gpsStatus_ == gpsStatus::ReInitialising || needsReInit){

        countValidMeasurements++;
        gpsMeasurementDeque.push_back(*rIterMeas);
      }
    }
    return countValidMeasurements;
}

bool ViGraph::addGpsMeasurements(GpsMeasurementDeque& gpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque, std::deque<StateId>* sids){

  if(gpsMeasurementDeque.size() == 0 || imuMeasurementDeque.size() == 0){
      return false;
  }

  if(!(imuMeasurementDeque.front().timeStamp <= gpsMeasurementDeque.front().timeStamp)){
    LOG(ERROR) << "IMU measurements not old enough for gps measurement. Can happen in beginning.";
    return false;
  }

  OKVIS_ASSERT_TRUE(Exception, (imuMeasurementDeque.front().timeStamp <= gpsMeasurementDeque.front().timeStamp), "IMU measurements not old enough for gps measurement");
  OKVIS_ASSERT_TRUE(Exception, (imuMeasurementDeque.back().timeStamp >= gpsMeasurementDeque.back().timeStamp), "GPS measurement not covered by IMU measurements");

  StateId sid; 
  if(sids != nullptr)
      sids->clear();

  if(gpsStatus_ != gpsStatus::Initialised){

      for(auto imuIter = imuMeasurementDeque.begin() ; imuIter != imuMeasurementDeque.end() ; ++imuIter){
          if(!gpsInitImuQueue_.empty() && (imuIter->timeStamp < gpsInitImuQueue_.back().timeStamp))
            continue;
          gpsInitImuQueue_.push_back(*imuIter);
        }
    }

  auto rIterStates = states_.rbegin();

  for(auto rIterMeas = gpsMeasurementDeque.rbegin(); rIterMeas != gpsMeasurementDeque.rend() ; rIterMeas++){

      while(rIterStates->second.timestamp > rIterMeas->timeStamp && rIterStates != states_.rend()){
          rIterStates++;
      }

      if(rIterStates == states_.rend()){
          break;
      }

      sid = rIterStates->first;
      if(sids != nullptr)
        sids->push_front(sid);

      states_.at(sid).gpsMode = gpsStatus_;

      switch(gpsStatus_){

        case gpsStatus::Off : 

          globCartesianFrame_.Reset(rIterMeas->measurement.latitude, rIterMeas->measurement.longitdue,
                                    rIterMeas->measurement.height);

          gpsInitMap_.insert({sid, *rIterMeas});
          LOG(INFO) << "[GPS] First measurements added. Starting GPS-VIO Initialisation.";
          gpsStatus_= gpsStatus::Idle;
          break;

        case gpsStatus::Idle: 

          gpsInitMap_.insert({sid, *rIterMeas});
          addGpsMeasurement(sid, *rIterMeas, imuMeasurementDeque);

          break;

        case gpsStatus::Initialising : 

          gpsInitMap_.insert({sid, *rIterMeas});
          addGpsMeasurement(sid, *rIterMeas, imuMeasurementDeque);

          break;

        case gpsStatus::Initialised : 
          addGpsMeasurement(sid, *rIterMeas, imuMeasurementDeque);
          break;

        case gpsStatus::ReInitialising : 

          gpsInitMap_.insert({sid, *rIterMeas});
          gpsReInitStates_.insert(sid);

          addGpsMeasurement(sid, *rIterMeas, imuMeasurementDeque);
          break;

        default:
            LOG(ERROR) << "[GPS] Wrong Mode Configured. Cannot add measurements.";
            return false;
        }

  }

  return true;
}

bool ViGraph::initializationStrategy(kinematics::Transformation& T_GW_est) {

  okvis::kinematics::Transformation T_GW_init;
  bool init_successful = false;
  double yaw_error = 100.0;
  switch(gpsStatus_){

    case gpsStatus::Off: 
      break;

    case gpsStatus::Idle: 

      checkForGpsInit(T_GW_init,gpsStates_, &yaw_error);
      if(yaw_error < 5.0){
        LOG(WARNING) << "Starting to add measurements to optimization now!";
        T_GW_est = T_GW_init;
        gpsStatus_ = gpsStatus::Initialising;
        init_successful = true;
      }
      break;

    case gpsStatus::Initialising : 
      gpsObservability_ = checkForGpsInit(T_GW_init,gpsStates_);
      if(gpsObservability_){
        setGpsExtrinsics(T_GW_init);
        gpsStatus_ = gpsStatus::Initialised;
        T_GW_init_ = T_GW_init;
        T_GW_est = T_GW_init;
        needsInitialAlignment_ = true;
        LOG(INFO) << "[GPS] GPS-VIO extrinsics have become observable. Initial Estimate: \n" << T_GW_init.T3x4();
      }
      break;

    case gpsStatus::Initialised: 
      break;

    case gpsStatus::ReInitialising: 
      gpsReInitialised_ = checkForGpsInit(T_GW_init,gpsReInitStates_);
      if(gpsReInitialised_){

        double distanceTravelled = 0.0; 
        kinematics::Transformation T_WS_i = states_.at(gpsDropoutId_).pose->estimate();
        auto iter = states_.find(gpsDropoutId_);
        ++iter;

        int num_steps = 0;
        for(; iter != states_.end(); ++iter) {
          kinematics::Transformation T_WS_j = iter->second.pose->estimate();
          const Eigen::Vector3d dsVec = (T_WS_j.r()-T_WS_i.r());
          const double ds = dsVec.norm();
          distanceTravelled += ds;
          T_WS_i = T_WS_j;
          num_steps++;
        }

        double rel_orientation_error = T_GW().q().angularDistance(T_GW_init.q()) / num_steps;
        double rel_orientation_error_budget = 0.03 + 0.004/sqrt(num_steps); 

        if(rel_orientation_error > rel_orientation_error_budget) {
          LOG(ERROR) << "Rejecting new T_GW due to angle error " << (rel_orientation_error * num_steps) * 180.0 / M_PI;
          LOG(ERROR) << "--- Allowed: " << (rel_orientation_error_budget * num_steps) * 180.0 / M_PI;
          gpsReInitialised_ = false;
        }
      } 

      if(gpsReInitialised_){
        needsFullAlignment_ = true;
        T_GW_init_ = T_GW_init;
        LOG(INFO) << "[GPS] GPS fully re-Initialised. Ready to apply global alignment.";
      }

      break;

    default:
      break;
  }

  return init_successful;

}

void ViGraph::addGpsInitFactors(){

  for(auto& init_measurement : gpsInitMap_){

    State& state = states_.at(init_measurement.first);
    for (auto& factor : state.GpsFactors) {
      factor.residualBlockId = problem_->AddResidualBlock(factor.errorTerm.get(), cauchyGpsLossFunctionPtr_.get(),
        state.pose->parameters(),state.speedAndBias->parameters(), state.T_GW->parameters());
    }
  }
}

void ViGraph::getLandmarkHessian(LandmarkId lmId, Eigen::Matrix3d& H) {
  OKVIS_ASSERT_TRUE_DBG(Exception,landmarks_.count(lmId) ,"parameter block not in map.");

  H.setZero();

  ::ceres::ResidualBlockId resBlock;
  KeypointIdentifier kid;

  for(auto iter = landmarks_.at(lmId).observations.begin(); iter != landmarks_.at(lmId).observations.end(); ++iter){

      kid = iter->first;
      resBlock = iter->second.residualBlockId;

      Eigen::Matrix<double,2,1> residuals;

      std::vector<double*> parametersVec;
      double* parameters[3];
      double* jacobians[3];
      Eigen::Matrix<double,2,7,Eigen::RowMajor> J0;
      Eigen::Matrix<double,2,4,Eigen::RowMajor> J1;
      Eigen::Matrix<double,2,7,Eigen::RowMajor> J2;
      jacobians[0]=J0.data();
      jacobians[1]=J1.data();
      jacobians[2]=J2.data();
      double* jacobiansMinimal[3];
      Eigen::Matrix<double,2,6,Eigen::RowMajor> J0min;
      Eigen::Matrix<double,2,3,Eigen::RowMajor> J1min;
      Eigen::Matrix<double,2,6,Eigen::RowMajor> J2min;
      jacobiansMinimal[0]=J0min.data();
      jacobiansMinimal[1]=J1min.data();
      jacobiansMinimal[2]=J2min.data();
      problem_->GetParameterBlocksForResidualBlock(resBlock,&parametersVec); 
      parameters[0] = parametersVec[0];
      parameters[1] = parametersVec[1];
      parameters[2] = parametersVec[2];

      landmarks_.at(lmId).observations.at(kid).errorTerm->EvaluateWithMinimalJacobians(
                  parameters,residuals.data(), jacobians, jacobiansMinimal);
      Eigen::Map< Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jl(jacobiansMinimal[1]);

      H+= Jl.transpose() * Jl;

  }
}

bool ViGraph::addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                            const uint64_t &frame_A_id, const uint64_t frame_B_id,
                                            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                            std::vector<float> sensorError, bool isLidar, const std::string robustFunction) {

  if((states_.count(StateId(frame_A_id)) == 0) || (states_.count(StateId(frame_B_id)) == 0)){
    LOG(ERROR) << "state outside of optimisation window";
    return false;
  }

  State & state_A = states_.at(StateId(frame_A_id));
  State & state_B = states_.at(StateId(frame_B_id));

  size_t non_zero_occupancy = 0;
  size_t non_zero_gradient = 0;
  double total_cost = 0.0;
  double max_residual = 0.0;
  size_t n_big_residuals = 0;
  float gradNorm = 0.0f;  
  float sumGradNorm = 0.0f;
  float maxGradNorm = 0.0f;
  float fieldVal = 0.0f;
  float sumFieldVal = 0.0f;
  float minFieldVal = 0.0f;
  float maxFieldVal = 0.0f;

  std::vector<::ceres::ResidualBlockId> resIds;

  for(uint64_t i = 0; i < pointCloud.size(); i++){
    SubmapAlignmentFactor newSubmapAlignmentFactor;
    Eigen::Vector3f pt = pointCloud.at(i);
    float sigma_i = sensorError.at(i);
    newSubmapAlignmentFactor.errorTerm.reset(new okvis::ceres::SubmapIcpError(*submap_ptr, pt.cast<double>(), sigma_i));
    ::ceres::ResidualBlockId rid;
    if(robustFunction == "Cauchy") {
      rid = problem_->AddResidualBlock(newSubmapAlignmentFactor.errorTerm.get(), cauchyLossFunctionPtr_.get(), state_A.pose->parameters(), state_B.pose->parameters());
      newSubmapAlignmentFactor.residualBlockId = rid;
    }
    else if(robustFunction == "Tukey") {
      if(isLidar) {
        rid = problem_->AddResidualBlock(newSubmapAlignmentFactor.errorTerm.get(), tukeyLidarLossFunctionPtr_.get(), state_A.pose->parameters(), state_B.pose->parameters());
      }
      else {
        rid = problem_->AddResidualBlock(newSubmapAlignmentFactor.errorTerm.get(), tukeyDepthLossFunctionPtr_.get(), state_A.pose->parameters(), state_B.pose->parameters());
      }
      newSubmapAlignmentFactor.residualBlockId = rid;
    }
    else {
      LOG(ERROR) << "Only Cauchy or Tukey robust functions are supported in the submap alignment.";
    }
    state_A.mapResIds.push_back(rid);
    state_A.submapReferenceLinks.push_back(newSubmapAlignmentFactor);
    state_B.mapResIds.push_back(rid);
    state_B.submapLinks.push_back(newSubmapAlignmentFactor);

    if(debugLidarResiduals_){
      resIds.push_back(rid);

      std::vector<double*> parameter_blocks;
      problem_->GetParameterBlocksForResidualBlock(rid, &parameter_blocks);

      Eigen::Matrix<double, 1, 1> residual;
      double *jacobians[2];
      Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J0;
      Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J1;
      jacobians[0] = J0.data();
      jacobians[1] = J1.data();
      static_cast<okvis::ceres::SubmapIcpError *>(newSubmapAlignmentFactor.errorTerm.get())->EvaluateWithMinimalJacobians(parameter_blocks.data(),
                                                                                              residual.data(), jacobians, nullptr);
      total_cost+=residual.norm();
      if(residual.norm() > max_residual) max_residual = residual.norm();
      if(residual.norm() > 3.0) n_big_residuals++;
      auto nonZeroFieldAndGradient = static_cast<okvis::ceres::SubmapIcpError*>(newSubmapAlignmentFactor.errorTerm.get())->nonZeroFieldAndGradient(parameter_blocks.data(), fieldVal, gradNorm);
      if(nonZeroFieldAndGradient.first) non_zero_occupancy++;
      if(nonZeroFieldAndGradient.second) non_zero_gradient++;

      sumGradNorm+=gradNorm;
      sumFieldVal+=fieldVal;
      if(fieldVal < minFieldVal) minFieldVal = fieldVal;
      if(fieldVal > maxFieldVal) maxFieldVal = fieldVal;
      if(gradNorm > maxGradNorm) maxGradNorm = gradNorm;
    }
  }
  if(debugLidarResiduals_) {
    double mean_cost = 0.0;
    if(pointCloud.size() > 0) mean_cost = total_cost / pointCloud.size();
    lidarDbgInfos_.push_back({frame_B_id, frame_A_id, pointCloud.size(), non_zero_occupancy, non_zero_gradient, mean_cost, max_residual, n_big_residuals,
                              float(sumFieldVal)/pointCloud.size(), minFieldVal, maxFieldVal,
                              float(sumGradNorm)/pointCloud.size(), maxGradNorm});

  }

  return true;
}

const Eigen::Vector4d &ViGraph::landmark(LandmarkId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  return landmarks_.at(id).hPoint->estimate();
}

bool ViGraph::getLandmark(LandmarkId landmarkId, MapPoint2& mapPoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  std::set<KeypointIdentifier> observations;
  const Landmark & landmark = landmarks_.at(landmarkId);
  for(auto observation : landmark.observations) {
    observations.insert(observation.first);
  }
  mapPoint = MapPoint2{landmarkId, landmark.hPoint->estimate(), observations,
      landmark.hPoint->initialized(), landmark.quality};
  return true;
}

size_t ViGraph::getLandmarks(MapPoints &landmarks) const
{
  for(auto& landmark : landmarks_) {
    std::set<KeypointIdentifier> observations;
    for(auto observation : landmark.second.observations) {
      observations.insert(observation.first);
    }
    landmarks[landmark.first] = MapPoint2{
        landmark.first, landmark.second.hPoint->estimate(), observations,
        landmark.second.hPoint->initialized(), landmark.second.quality};
  }
  return landmarks.size();
}

bool ViGraph::landmarkExists(LandmarkId landmarkId) const
{
  return landmarks_.count(landmarkId) != 0;
}

bool ViGraph::setLandmark(LandmarkId id, const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  auto& landmark = landmarks_.at(id).hPoint;
  landmark->setEstimate(homogeneousPoint);
  landmark->setInitialized(initialised);
  return true;
}
bool ViGraph::setLandmark(LandmarkId id, const Eigen::Vector4d &homogeneousPoint)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  auto& landmark = landmarks_.at(id).hPoint;
  landmark->setEstimate(homogeneousPoint);
  return true;
}

bool ViGraph::areLandmarksInFrontOfCameras() const
{
  for (const auto &obs : observations_) {
    Eigen::Vector4d hp_W = landmark(obs.second.landmarkId);
    kinematics::Transformation T_WS, T_SCi;
    const StateId stateId(obs.first.frameId);
    const State &state = states_.at(stateId);
    T_WS = state.pose->estimate();
    T_SCi = state.extrinsics.at(obs.first.cameraIndex)->estimate();
    kinematics::Transformation T_WCi = T_WS * T_SCi;
    Eigen::Vector4d pos_Ci = T_WCi.inverse() * hp_W;
    if (pos_Ci[2] * pos_Ci[3] < 0.0) {
      return false;
    }
    if (fabs(pos_Ci[3]) > 1.0e-16) {
      if (pos_Ci[2] / pos_Ci[3] < 0.05) {
        return false;
      }
    }
  }
  return true;
}

const kinematics::TransformationCacheless & ViGraph::pose(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).pose->estimate();
}

const SpeedAndBias &ViGraph::speedAndBias(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).speedAndBias->estimate();
}

const kinematics::TransformationCacheless & ViGraph::T_GW() const
{
  return states_.begin()->second.T_GW->estimate();
}

const kinematics::TransformationCacheless & ViGraph::T_GW(StateId id) const
{
  OKVIS_ASSERT_TRUE(Exception, states_.count(id), "state does not exists")
  return states_.at(id).T_GW->estimate();
}

const kinematics::TransformationCacheless &ViGraph::extrinsics(StateId id, uchar camIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(id).extrinsics.at(camIdx), "extrinsics do not exists")
      return states_.at(id).extrinsics.at(camIdx)->estimate();
}

bool ViGraph::isKeyframe(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).isKeyframe;
}

Time ViGraph::timestamp(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).timestamp;
}

StateId ViGraph::currentStateId() const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, !states_.empty(), "no states exist")
  return states_.rbegin()->first;
}

StateId ViGraph::stateIdByAge(size_t age) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.size()>age, "too few states exist")
  auto iter = states_.rbegin();
  for(size_t k=0; k<age; k++) {
    iter++;
  }
  return iter->first;
}

bool ViGraph::setKeyframe(StateId id, bool isKeyframe)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).isKeyframe = isKeyframe;
  return true;
}

bool ViGraph::setPose(StateId id, const kinematics::TransformationCacheless &pose)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).pose->setEstimate(pose);
  return true;
}

bool ViGraph::setSpeedAndBias(StateId id, const SpeedAndBias &speedAndBias)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).speedAndBias->setEstimate(speedAndBias);
  return true;
}

bool ViGraph::setExtrinsics(StateId id, uchar camIdx,
                            const kinematics::TransformationCacheless &extrinsics) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(id).extrinsics.at(camIdx), "extrinsics do not exists")
  states_.at(id).extrinsics.at(camIdx)->setEstimate(extrinsics);
  return true;
}

bool ViGraph::setExtrinsicsVariable()
{
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {
    problem_->SetParameterBlockVariable(states_.begin()->second.extrinsics.at(i)->parameters());
  }
  return true;
}

bool ViGraph::softConstrainExtrinsics(double posStd, double rotStd)
{
  if(states_.begin()->second.extrinsicsPriors.size() == cameraParametersVec_.size())
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {
    problem_->RemoveResidualBlock(states_.begin()->second.extrinsicsPriors.at(i).residualBlockId);
  }
  states_.begin()->second.extrinsicsPriors.resize(cameraParametersVec_.size());
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {

    PosePrior& extrinsicsPrior = states_.begin()->second.extrinsicsPriors.at(i);
    extrinsicsPrior.errorTerm.reset(
          new ceres::PoseError(
            states_.begin()->second.extrinsics.at(i)->estimate(), posStd*posStd, rotStd*rotStd));
    extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
              extrinsicsPrior.errorTerm.get(), nullptr,
          states_.begin()->second.extrinsics.at(i)->parameters());

  }
  return true;
}

void ViGraph::updateLandmarks()
{
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ++it) {
    Eigen::Vector4d hp_W = it->second.hPoint->estimate();
    const size_t num = it->second.observations.size();
    bool isInitialised = false;
    double quality = 0.0;
    bool behind = false;
    double best_err = 1.0e12;
    Eigen::Vector4d best_pos(0.0,0.0,0.0,0.0);
    if(num>0){
      Eigen::Array<double, 3, Eigen::Dynamic> dirs(3,num);
      int o=0;

      for(const auto& observation : it->second.observations) {
        kinematics::Transformation T_WS, T_SCi;
        const StateId stateId(observation.first.frameId);
        const State & state = states_.at(stateId);
        T_WS = state.pose->estimate();
        T_SCi = state.extrinsics.at(observation.first.cameraIndex)->estimate();
        kinematics::Transformation T_WCi = T_WS*T_SCi;
        Eigen::Vector4d pos_Ci = T_WCi.inverse()*hp_W;

        if(fabs(pos_Ci[3])>1.0e-12) {
          pos_Ci = pos_Ci/pos_Ci[3];
        }
        Eigen::Vector3d dir_W = (T_WCi.C()*pos_Ci.head<3>()).normalized();
        if(pos_Ci[2]<0.1) {
          behind = true;
        }
        if(pos_Ci[2] < 0.0) {
          dir_W = -dir_W; 
        }

        Eigen::Vector2d err;
        double* params[3];
        params[0] = state.pose->parameters();
        params[1] = it->second.hPoint->parameters();
        params[2] = state.extrinsics.at(observation.first.cameraIndex)->parameters();
        observation.second.errorTerm->Evaluate(params, err.data(), nullptr);
        const double err_norm = err.norm();

        if(err_norm>2.5) {

          continue;
        }

        if(err_norm < best_err && pos_Ci.norm()>0.0001) {

          const double dist = std::max(0.1,pos_Ci.norm());
          best_pos.head<3>() = T_WCi.r() + dist*dir_W;
          best_pos[3]=1.0;
          best_err = err_norm;
        }

        dirs.col(o) = dir_W;
        ++o;
      }
      Eigen::Array<double, 3, Eigen::Dynamic> dirso(3,o);
      dirso = dirs.topLeftCorner(3,o);
      Eigen::Vector3d std_dev =
        ((dirso.colwise() - dirso.rowwise().mean()).square().rowwise().sum()).sqrt();
      quality = std_dev.norm();
      if(quality > 0.04) {
        isInitialised = true;
      } else {
        if(behind && best_pos.norm()>1.0e-12) {

          it->second.hPoint->setEstimate(best_pos);
        }
      }
    }

    it->second.hPoint->setInitialized(isInitialised);
    it->second.quality = quality;
  }
}

#ifdef USE_OPENMP
void ViGraph::optimise(int maxIterations, int numThreads, bool verbose)
#else

void ViGraph::optimise(int maxIterations, int , bool verbose)
#warning openmp not detected, your system may be slower than expected
#endif
{

#ifdef USE_OPENMP
  options_.num_threads = int(numThreads);
#endif
  options_.max_num_iterations = int(maxIterations);

  if (verbose) {
    options_.minimizer_progress_to_stdout = true;
  } else {
    options_.minimizer_progress_to_stdout = false;
  }

  ::ceres::Solve(options_, problem_.get(), &summary_);

  if (verbose) {
    LOG(INFO) << summary_.FullReport();
  }
}

bool ViGraph::setOptimisationTimeLimit(double timeLimit, int minIterations)
{
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {

      ceresCallback_->setMinimumIterations(options_.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
          new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    options_.callbacks.push_back(ceresCallback_.get());
    return true;
  }

  return true;
}

int ViGraph::cleanUnobservedLandmarks(std::map<LandmarkId, std::set<KeypointIdentifier> > *removed)
{
  int ctr = 0;
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
    const auto& lm = it->second;
    if(lm.observations.size()<=1) {
      if(removed) {
        (*removed)[it->first] = std::set<KeypointIdentifier>();
      }
      if(lm.observations.size()==1) {
        if(removed) {
          removed->at(it->first).insert(lm.observations.begin()->first);
        }
        removeObservation(lm.observations.begin()->first);
      }
      problem_->RemoveParameterBlock(lm.hPoint->parameters());
      it = landmarks_.erase(it);
      ctr++;
    }
    else {
      ++it;
    }
  }
  return ctr;
}

void ViGraph::writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix)
{

  std::string csvFileName = csvFilePrefix + "-lidar-info.csv";
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success = csvFile.good();
  if (!success) {
    return;
  }

  LOG(INFO) << "[DEBUG INFO Lidar Alignment] Dumping debug info to csv file.";

  csvFile << "# frame id, reference frame id, n_residual, n_nonzro_occ, n_non_zero_grad, mean res., max res., n_big_residuals, mean occ, min occ, max occ, mean grad norm, max grad norm" << std::endl;

  for (auto dbgInfo : lidarDbgInfos_)
  {
    csvFile << dbgInfo.frameId << ", " << dbgInfo.referenceId
            << ", " << dbgInfo.n_residuals
            << ", " << dbgInfo.n_nonzro_occ
            << ", " << dbgInfo.n_non_zero_grad
            << ", " << dbgInfo.mean_res
            << ", " << dbgInfo.max_residual
            << ", " << dbgInfo.n_big_residuals
            << ", " << dbgInfo.meanFieldVal
            << ", " << dbgInfo.minFieldVal
            << ", " << dbgInfo.maxFieldVal 
            << ", " << dbgInfo.meanGradNorm
            << ", " << dbgInfo.maxGradNorm << std::endl;    
  }
  csvFile.close();
}

}  
