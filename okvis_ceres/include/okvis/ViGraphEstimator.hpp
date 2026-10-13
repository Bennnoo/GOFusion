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
 * @file okvis/ViGraphEstimator.hpp
 * @brief Header file for the ViGraphEstimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_VIGRAPHESTIMATOR_HPP_
#define INCLUDE_OKVIS_VIGRAPHESTIMATOR_HPP_

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/cost_function.h>
#include <ceres/crs_matrix.h>
#include <ceres/evaluation_callback.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/ordered_groups.h>
#include <ceres/problem.h>
#include <ceres/product_manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <ceres/version.h>
#pragma GCC diagnostic pop

#include <opencv2/core.hpp>

#include <okvis/ViGraph.hpp>

namespace okvis {

class ViGraphEstimator : public ViGraph
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  friend class ViSlamBackend;
  friend class Component;

  ViGraphEstimator() {}

  ~ViGraphEstimator() {}

  bool removeAllObservations(StateId stateId);

  bool eliminateStateByImuMerge(StateId stateId, StateId refId);

  bool mergeLandmark(LandmarkId fromId, LandmarkId intoId,
                     std::map<StateId,MultiFramePtr> &multiFrames);

  bool freezePosesUntil(StateId stateId, bool removeInCeres = false);

  bool unfreezePosesFrom(StateId stateId);

  bool freezeSpeedAndBiasesUntil(StateId stateId, bool removeInCeres = false);

  bool unfreezeSpeedAndBiasesFrom(StateId stateId);

  bool freezeExtrinsicsUntil(StateId stateId, bool removeInCeres = false);

  bool unfreezeExtrinsicsFrom(StateId stateId);

  struct PoseGraphEdge {
    std::shared_ptr<ceres::TwoPoseGraphError> poseGraphError; 
    StateId referenceId; 
    StateId otherId; 
  };

  bool convertToPoseGraphMst(
      const std::set<StateId>& states, const std::set<StateId> &statesToConsider,
      std::vector<PoseGraphEdge>* createdPoseGraphEdges = nullptr,
      std::vector<std::pair<StateId, StateId> > *removedTwoPoseErrors = nullptr,
      std::vector<KeypointIdentifier> *removedObservations = nullptr);

  bool obtainPoseGraphMst(std::vector<PoseGraphEdge>& poseGraphEdges);

  bool addExternalTwoPoseLink(const std::shared_ptr<ceres::TwoPoseGraphErrorConst> &twoPoseError,
                         StateId referenceId, StateId otherId);

  bool removeTwoPoseConstLinks(StateId convertStateId);

  bool removeTwoPoseConstLink(StateId pose_i, StateId pose_j);

  int convertToObservations(
      StateId convertStateId,
      std::set<LandmarkId> *createdLandmarks = nullptr,
      std::set<StateId> *connectedStates = nullptr,
      std::vector<ceres::TwoPoseGraphError::Observation> *observations = nullptr);

  bool removeSpeedAndBiasPrior(StateId stateId);

  void clear();

protected:

  int buildMst(const std::set<StateId>& states, bool keyframesOnly=true);

  bool isSynched(const ViGraphEstimator &other) const;

  bool isObservationlessSynched(const ViGraphEstimator & other) const;

  void checkSynchedStates(const ViGraphEstimator & other) const; 

  std::vector<uint64_t> mstFrameIds_; 
  std::map<uint64_t, size_t> mstGraphIdxs_; 
  std::vector<std::pair<int,int>> mstEdges_; 
};

}  

#endif 
