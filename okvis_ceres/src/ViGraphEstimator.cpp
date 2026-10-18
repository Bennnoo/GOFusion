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
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

/**
 * @file ViGraphEstimator.cpp
 * @brief Implements the ViGraphEstimator class. This does much of the backend work.
 * @author Stefan Leutenegger
 */

#include <algorithm>

#include <okvis/ceres/TwoPoseExtrinsicsGraphError.hpp>
#include <okvis/ViGraphEstimator.hpp>
#include <okvis/MstGraph.hpp>

namespace okvis {

bool ViGraphEstimator::removeAllObservations(StateId stateId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  std::map<KeypointIdentifier, Observation> observations = states_.at(stateId).observations;
  for(auto observation : observations) {
    removeObservation(observation.first);
  }
  return true;
}

bool ViGraphEstimator::eliminateStateByImuMerge(StateId stateId, StateId refId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")

  auto relativePoseLinks = states_.at(stateId).relativePoseLinks;
  for(auto relativePoseLink : relativePoseLinks) {
    removeRelativePoseConstraint(relativePoseLink.second.state0, relativePoseLink.second.state1);
  }

  const State & state = states_.at(stateId);
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(stateId).twoPoseConstLinks.size()==0,
                        "two pose const links present at state ID=" << stateId.value())
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(stateId).twoPoseLinks.size()==0,
                        "two pose links present at state ID=" << stateId.value())
  auto iterPrevious = states_.find(stateId);
  OKVIS_ASSERT_TRUE_DBG(Exception, iterPrevious!=states_.begin(), "cannot eliminate first state")
  iterPrevious--;
  auto iterNext = states_.find(stateId);
  iterNext++;
  OKVIS_ASSERT_TRUE_DBG(Exception, iterNext!=states_.end(), "cannot eliminate last state")
  State & previousState = iterPrevious->second;
  State & nextState = iterNext->second;

  ImuLink & previousImuLink = previousState.nextImuLink;

  kinematics::Transformation T_WS = state.pose->estimate();
  SpeedAndBias speedAndBias = state.speedAndBias->estimate();

  if(imuParametersVec_.at(0).use) {

    std::static_pointer_cast<ceres::ImuError>(previousImuLink.errorTerm)
      ->append(T_WS,
               speedAndBias,
               std::static_pointer_cast<ceres::ImuError>(state.nextImuLink.errorTerm)
                 ->imuMeasurements(),
               state.nextImuLink.errorTerm->t1());
  } else {

    previousImuLink.errorTerm->setT1(state.nextImuLink.errorTerm->t1());
  }

  problem_->RemoveResidualBlock(state.nextImuLink.residualBlockId);
  problem_->RemoveResidualBlock(previousImuLink.residualBlockId);

  problem_->RemoveParameterBlock(state.pose->parameters());  
  problem_->RemoveParameterBlock(state.speedAndBias->parameters());  

  previousImuLink.residualBlockId = problem_->AddResidualBlock(
      previousImuLink.errorTerm.get(), nullptr,
      previousState.pose->parameters(),  previousState.speedAndBias->parameters(),
      nextState.pose->parameters(),  nextState.speedAndBias->parameters());

  OKVIS_ASSERT_TRUE_DBG(Exception,
                        previousState.pose->timestamp() == previousImuLink.errorTerm->t0(),
                        "inconsistent timestamps")
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        nextState.pose->timestamp() == previousImuLink.errorTerm->t1(),
                        "inconsistent timestamps")

  nextState.previousImuLink = previousImuLink;

  for (size_t i = 0; i < state.extrinsics.size(); ++i) {
    if (state.extrinsics.at(i)->fixed()) {
      continue;  
    }

    if (state.extrinsics.at(i)->id() == previousState.extrinsics.at(i)->id()) {
      continue;
    }

    OKVIS_THROW(Exception, "varying extrinsics not supported")
  }

  AnyState& anyState = anyState_.at(stateId);
  const StateId keyframeId = refId;
  OKVIS_ASSERT_TRUE(Exception, keyframeId.value(), "no keyframe!")
  anyState.keyframeId = keyframeId;
  State& otherstate = states_.at(keyframeId);
  kinematics::Transformation T_Sk_W = otherstate.pose->estimate().inverse();
  anyState.T_Sk_S = T_Sk_W * states_[stateId].pose->estimate();
  anyState.v_Sk = T_Sk_W.C() * otherstate.speedAndBias->estimate().head<3>();

  OKVIS_ASSERT_TRUE(Exception,
                    states_.at(stateId).relativePoseLinks.size() == 0,
                    "relative pose links present")
  states_.erase(stateId);
  gpsStates_.erase(stateId); 
  gpsReInitStates_.erase(stateId);
  gpsInitMap_.erase(stateId);

  return true;
}

bool ViGraphEstimator::mergeLandmark(LandmarkId fromId, LandmarkId intoId,
                                     std::map<StateId, MultiFramePtr> &multiFrames)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(fromId), "Landmark ID not found")
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(intoId), "Landmark ID not found")

  Landmark & fromLandmark = landmarks_.at(fromId);
  std::map<KeypointIdentifier, Observation> observations = fromLandmark.observations;
  for(auto observation : observations) {

    bool useLoss = false;
    if(problem_->GetLossFunctionForResidualBlock(observation.second.residualBlockId)) {
      useLoss = true;
    }

    removeObservation(observation.first);

    State & state = states_.at(StateId(observation.first.frameId));
    observation.second.residualBlockId = problem_->AddResidualBlock(
        observation.second.errorTerm.get(), useLoss ? cauchyLossFunctionPtr_.get() : nullptr,
        state.pose->parameters(), landmarks_.at(intoId).hPoint->parameters(),
        state.extrinsics.at(observation.first.cameraIndex)->parameters());
    observation.second.landmarkId = intoId;

    observations_[observation.first] = observation.second;
    landmarks_.at(intoId).observations[observation.first] = observation.second;
    state.observations[observation.first] = observation.second;

    multiFrames.at(StateId(observation.first.frameId))->setLandmarkId(
          observation.first.cameraIndex, observation.first.keypointIndex, intoId.value());
  }

  OKVIS_ASSERT_TRUE_DBG(Exception, fromLandmark.observations.size() == 0, "observations present")
  removeLandmark(fromId);

  return true;
}

bool ViGraphEstimator::freezePosesUntil(StateId stateId, bool removeInCeres)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  bool first = true;
  for(auto iter = states_.find(stateId); ; --iter) {
    if(iter->second.pose->fixed()) {
      if(removeInCeres) {
        if(problem_->HasParameterBlock(iter->second.pose->parameters())) {
          problem_->RemoveParameterBlock(iter->second.pose->parameters());
        }
      }
      break;
    }
    iter->second.pose->setFixed(true);
    if(first || !removeInCeres) {
      problem_->SetParameterBlockConstant(iter->second.pose->parameters());
      first = false;
    } else {
      if(problem_->HasParameterBlock(iter->second.pose->parameters())) {
        problem_->RemoveParameterBlock(iter->second.pose->parameters());
      }
    }
    if(iter==states_.begin()) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::unfreezePosesFrom(StateId stateId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  for(auto riter = states_.rbegin(); riter != states_.rend(); ++riter) {
    riter->second.pose->setFixed(false);
    problem_->SetParameterBlockVariable(riter->second.pose->parameters());
    if(riter->first == stateId) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::freezeSpeedAndBiasesUntil(StateId stateId, bool removeInCeres)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  bool first = true;
  for(auto iter = states_.find(stateId); ; --iter) {
    if(iter->second.speedAndBias->fixed()) {
      if(removeInCeres) {
        if(problem_->HasParameterBlock(iter->second.speedAndBias->parameters())) {
          problem_->RemoveParameterBlock(iter->second.speedAndBias->parameters());
        }
      }
      break;
    }
    iter->second.speedAndBias->setFixed(true);
    if(first || !removeInCeres) {
      problem_->SetParameterBlockConstant(iter->second.speedAndBias->parameters());
      first = false;
    } else {
      if(problem_->HasParameterBlock(iter->second.speedAndBias->parameters())) {
        problem_->RemoveParameterBlock(iter->second.speedAndBias->parameters());
      }
    }
    if(iter==states_.begin()) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::unfreezeSpeedAndBiasesFrom(StateId stateId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  for(auto riter = states_.rbegin(); riter != states_.rend(); ++riter) {
    riter->second.speedAndBias->setFixed(false);
    problem_->SetParameterBlockVariable(riter->second.speedAndBias->parameters());
    if(riter->first == stateId) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::freezeExtrinsicsUntil(StateId stateId, bool removeInCeres)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  OKVIS_ASSERT_TRUE(Exception, !removeInCeres, "not supported")
  for(auto iter = states_.find(stateId); ; --iter) {
    for(size_t i=0; i<iter->second.extrinsics.size(); ++i) {
      if(iter->second.extrinsics.at(i)->fixed()) {
        break;
      }
      iter->second.extrinsics.at(i)->setFixed(true);
      problem_->SetParameterBlockConstant(iter->second.extrinsics.at(i)->parameters());
    }
    if(iter==states_.begin()) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::unfreezeExtrinsicsFrom(StateId stateId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId), "State ID not found")
  for(auto riter = states_.rbegin(); riter != states_.rend(); ++riter) {
    for(size_t i=0; i<riter->second.extrinsics.size(); ++i) {
      riter->second.extrinsics.at(i)->setFixed(false);
      problem_->SetParameterBlockVariable(riter->second.extrinsics.at(i)->parameters());
    }
    if(riter->first == stateId) {
      break;
    }
  }
  return true;
}

bool ViGraphEstimator::convertToPoseGraphMst(
    const std::set<StateId> & states,
    const std::set<StateId> & statesToConsider,
    std::vector<PoseGraphEdge> * createdPoseGraphEdges,
    std::vector<std::pair<StateId,StateId>> * removedTwoPoseErrors,
    std::vector<KeypointIdentifier> * removedObservations)
{

  buildMst(statesToConsider);
  if(mstEdges_.empty()) {
    return false; 
  }

  std::map<uint64_t, size_t> numEdges;
  for(auto edge : mstEdges_) {
    const uint64_t idFirst = mstFrameIds_.at(size_t(edge.first));
    const uint64_t idSecond = mstFrameIds_.at(size_t(edge.second));
    if(!numEdges.count(idFirst)) {
      numEdges[idFirst] = 1;
    } else {
      numEdges.at(idFirst)++;
    }
    if(!numEdges.count(idSecond)) {
      numEdges[idSecond] = 1;
    } else {
      numEdges.at(idSecond)++;
    }
  }
  std::vector<std::pair<uint64, uint64>> edgesToCreate;

  for(auto edge : mstEdges_) {
    const uint64_t idFirst = mstFrameIds_.at(size_t(edge.first));
    const uint64_t idSecond = mstFrameIds_.at(size_t(edge.second));
    if(states.count(StateId(idFirst)) || states.count(StateId(idSecond))) {
      edgesToCreate.push_back(edge);
    }
  }

  const StateId idNewest(mstFrameIds_.back());
  const StateId idOldest(mstFrameIds_.front());
  if(states.count(idOldest)) {
    if(covisibilities(idNewest, idOldest) >= 2 && idNewest!=idOldest) {
      bool alreadyCreated = false;
      for(auto edge : edgesToCreate) {
        if(edge.first == mstGraphIdxs_.at(idOldest.value())
           && edge.second == mstGraphIdxs_.at(idNewest.value())) {
          alreadyCreated = true;
          break;
        }
        if(edge.first == mstGraphIdxs_.at(idNewest.value())
           && edge.second == mstGraphIdxs_.at(idOldest.value())) {
          alreadyCreated = true;
          break;
        }
      }
      if(!alreadyCreated) {
        edgesToCreate.push_back(std::make_pair(
                                  mstGraphIdxs_.at(idOldest.value()),
                                  mstGraphIdxs_.at(idNewest.value())));
        if(!numEdges.count(idOldest.value())) {
          numEdges[idOldest.value()] = 1;
        } else {
          numEdges.at(idOldest.value())++;
        }
        if(!numEdges.count(idNewest.value())) {
          numEdges[idNewest.value()] = 1;
        } else {
          numEdges.at(idNewest.value())++;
        }
      }
    }
  }

  for(auto edge : edgesToCreate) {

    TwoPoseLink twoPoseLink;
    StateId referenceStateId =
        StateId(std::min(mstFrameIds_.at(edge.first), mstFrameIds_.at(edge.second)));
    StateId otherStateId =
        StateId(std::max(mstFrameIds_.at(edge.first), mstFrameIds_.at(edge.second)));
    if (cameraParametersVec_.at(0).online_calibration.do_extrinsics) {
      twoPoseLink.errorTerm.reset(
        new ceres::TwoPoseExtrinsicsGraphError(referenceStateId, otherStateId,
                                               cameraParametersVec_.size()));
    } else {
      twoPoseLink.errorTerm.reset(
        new ceres::TwoPoseStandardGraphError(referenceStateId, otherStateId,
                                             cameraParametersVec_.size()));
    }
    twoPoseLink.state0 = referenceStateId;
    twoPoseLink.state1 = otherStateId;
    State & referenceState = states_.at(referenceStateId);
    State & otherState = states_.at(otherStateId);
    OKVIS_ASSERT_TRUE_DBG(Exception, referenceState.isKeyframe, "state must be KF")
    OKVIS_ASSERT_TRUE_DBG(Exception, otherState.isKeyframe, "state must be KF")

    if(referenceState.twoPoseLinks.count(otherStateId)) {
      std::vector<ceres::TwoPoseGraphError::Observation> obs;
      std::vector<KeypointIdentifier> dup;
      referenceState.twoPoseLinks.at(otherStateId).errorTerm->convertToReprojectionErrors(obs, dup);
      for(auto ob : obs) {
        if(ob.isDuplication) {
          ob.reprojectionError->setInformation(ob.reprojectionError->information());
        }
        twoPoseLink.errorTerm->addObservation(ob.keypointIdentifier, ob.reprojectionError,
                                              ob.lossFunction, ob.pose, ob.hPoint, ob.extrinsics,
                                              ob.isDuplication);
      }

      problem_->RemoveResidualBlock(referenceState.twoPoseLinks.at(otherStateId).residualBlockId);

      referenceState.twoPoseLinks.erase(otherStateId);
      otherState.twoPoseLinks.erase(referenceStateId);
      if(removedTwoPoseErrors) {
        removedTwoPoseErrors->push_back(std::pair<StateId,StateId>(referenceStateId, otherStateId));
      }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            referenceState.twoPoseLinks.count(otherStateId) == 0,
                            "two pose links still present")
    }

    bool keepReferenceState = (numEdges.at(referenceStateId.value()) > 1);
    bool keepOtherState = (numEdges.at(otherStateId.value()) > 1);
    if(!states.count(referenceStateId)) { 
      keepReferenceState = true;
    }
    if(!states.count(otherStateId)) { 
      keepOtherState = true;
    }

    std::map<KeypointIdentifier, Observation> referenceObservations = referenceState.observations;
    std::set<uint64_t> referenceLandmarks;
    for(auto observation : referenceObservations) {
      referenceLandmarks.insert(observation.second.landmarkId.value());
    }
    std::map<KeypointIdentifier, Observation> otherObservations = otherState.observations;
    std::set<uint64_t> otherLandmarks;
    for(auto observation : otherObservations) {
      otherLandmarks.insert(observation.second.landmarkId.value());
    }
    std::set<uint64_t> landmarksConsidered1;
    std::set_intersection(referenceLandmarks.begin(), referenceLandmarks.end(),
                          otherLandmarks.begin(), otherLandmarks.end(),
                          std::inserter(landmarksConsidered1,landmarksConsidered1.begin()));

    std::vector<std::pair<double, LandmarkId>> qualities;
    for(auto lm : landmarksConsidered1) {
      qualities.push_back(std::pair<double, LandmarkId>(
                            landmarks_.at(LandmarkId(lm)).quality, LandmarkId(lm)));
    }

    std::set<uint64_t> landmarksConsidered;
    if(qualities.size()>=10) {
      std::sort(qualities.begin(),qualities.end());
    }
    for(auto q : qualities) {
      landmarksConsidered.insert(q.second.value());
    }

    for(auto observation : referenceObservations) {
      Landmark & landmark = landmarks_.at(observation.second.landmarkId);

      if(landmarksConsidered.count(observation.second.landmarkId.value())==0) {
        if(!keepReferenceState) {
          removeObservation(observation.first);
          removedObservations->push_back(observation.first);
        }
        continue;
      }
      if(keepReferenceState) {

        observation.second.errorTerm->setInformation(
              observation.second.errorTerm->information());
      }
      twoPoseLink.errorTerm->addObservation(
            observation.first, observation.second.errorTerm, cauchyLossFunctionPtr_.get(),
            referenceState.pose, landmark.hPoint,
            referenceState.extrinsics.at(observation.first.cameraIndex), keepReferenceState);
      if(!keepReferenceState) {

        removeObservation(observation.first);
        removedObservations->push_back(observation.first);
      }
    }
    for(auto observation : otherObservations) {
      Landmark & landmark = landmarks_.at(observation.second.landmarkId);

      if(landmarksConsidered.count(observation.second.landmarkId.value())==0) {
        if(!keepOtherState) {
          removeObservation(observation.first);
          removedObservations->push_back(observation.first);
        }
        continue;
      }
      if(keepOtherState) {

        observation.second.errorTerm->setInformation(
              observation.second.errorTerm->information());
      }
      twoPoseLink.errorTerm->addObservation(
            observation.first, observation.second.errorTerm, cauchyLossFunctionPtr_.get(),
            otherState.pose, landmark.hPoint,
            otherState.extrinsics.at(observation.first.cameraIndex), keepOtherState);
      if(!keepOtherState) {

        removeObservation(observation.first);
        removedObservations->push_back(observation.first);
      }
    }

    if(landmarksConsidered.size()>0) {
      twoPoseLink.errorTerm->compute();

      if (cameraParametersVec_.at(0).online_calibration.do_extrinsics) {
        std::vector<double *> pars;
        pars.push_back(referenceState.pose->parameters());
        pars.push_back(otherState.pose->parameters());
        for (const auto &extrinsics : referenceState.extrinsics) {
          pars.push_back(extrinsics->parameters());
        }
        twoPoseLink.residualBlockId = problem_->AddResidualBlock(
          std::static_pointer_cast<ceres::TwoPoseExtrinsicsGraphError>(twoPoseLink.errorTerm).get(),
          nullptr,pars);
      } else {
        twoPoseLink.residualBlockId = problem_->AddResidualBlock(
          std::static_pointer_cast<ceres::TwoPoseStandardGraphError>(twoPoseLink.errorTerm).get(),
          nullptr,
          referenceState.pose->parameters(),
          otherState.pose->parameters());
      }

      OKVIS_ASSERT_TRUE_DBG(Exception,
                            referenceState.twoPoseLinks.count(otherStateId) == 0,
                            "two pose links present, ID " << referenceStateId.value() << "<->"
                                                          << otherStateId.value())
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            otherState.twoPoseLinks.count(referenceStateId) == 0,
                            "two pose links present, ID " << referenceStateId.value() << "<->"
                                                          << otherStateId.value())

      referenceState.twoPoseLinks[otherStateId] = twoPoseLink;
      otherState.twoPoseLinks[referenceStateId] = twoPoseLink;

      numEdges.at(referenceStateId.value())--;
      numEdges.at(otherStateId.value())--;

      if(createdPoseGraphEdges) {
        createdPoseGraphEdges->push_back(
              PoseGraphEdge{twoPoseLink.errorTerm, referenceStateId, otherStateId});
      }
    } else {

      numEdges.at(referenceStateId.value())--;
      numEdges.at(otherStateId.value())--;
    }
  }

  return true;
}

bool ViGraphEstimator::obtainPoseGraphMst(
    std::vector<ViGraphEstimator::PoseGraphEdge> &poseGraphEdges)
{

  buildMst(visibleFrames_, false);
  OKVIS_ASSERT_TRUE_DBG(Exception, !mstEdges_.empty() ,"MST not created properly")

  std::map<KeypointIdentifier, double> weights;

  std::map<uint64_t, size_t> numEdges;
  for(auto edge : mstEdges_) {
    const uint64_t idFirst = mstFrameIds_.at(size_t(edge.first));
    const uint64_t idSecond = mstFrameIds_.at(size_t(edge.second));
    if(!numEdges.count(idFirst)) {
      numEdges[idFirst] = 1;
    } else {
      numEdges.at(idFirst)++;
    }
    if(!numEdges.count(idSecond)) {
      numEdges[idSecond] = 1;
    } else {
      numEdges.at(idSecond)++;
    }
  }
  std::vector<std::pair<uint64, uint64>> edgesToCreate;
  for(auto edge : mstEdges_) {
    const uint64_t idFirst = mstFrameIds_.at(size_t(edge.first));
    const uint64_t idSecond = mstFrameIds_.at(size_t(edge.second));
    if(visibleFrames_.count(StateId(idFirst)) || visibleFrames_.count(StateId(idSecond))) {
      edgesToCreate.push_back(edge);
    }
  }

  for(auto edge : edgesToCreate) {

    TwoPoseLink twoPoseLink;
    StateId referenceStateId =
        StateId(std::min(mstFrameIds_.at(edge.first), mstFrameIds_.at(edge.second)));
    StateId otherStateId =
        StateId(std::max(mstFrameIds_.at(edge.first), mstFrameIds_.at(edge.second)));
    if (cameraParametersVec_.at(0).online_calibration.do_extrinsics) {
      twoPoseLink.errorTerm.reset(
        new ceres::TwoPoseExtrinsicsGraphError(referenceStateId, otherStateId,
                                               cameraParametersVec_.size()));
    } else {
      twoPoseLink.errorTerm.reset(
        new ceres::TwoPoseStandardGraphError(referenceStateId, otherStateId,
                                             cameraParametersVec_.size()));
    }
    State & referenceState = states_.at(referenceStateId);
    State & otherState = states_.at(otherStateId);

    bool keepReferenceState = (numEdges.at(referenceStateId.value())>1);
    bool keepOtherState = (numEdges.at(otherStateId.value())>1);
    if(!visibleFrames_.count(referenceStateId)) { 
      keepReferenceState = true;
    }
    if(!visibleFrames_.count(otherStateId)) { 
      keepOtherState = true;
    }

    std::map<KeypointIdentifier, Observation> referenceObservations = referenceState.observations;
    std::set<uint64_t> referenceLandmarks;
    for(auto observation : referenceObservations) {
      referenceLandmarks.insert(observation.second.landmarkId.value());
    }
    std::map<KeypointIdentifier, Observation> otherObservations = otherState.observations;
    std::set<uint64_t> otherLandmarks;
    for(auto observation : otherObservations) {
      otherLandmarks.insert(observation.second.landmarkId.value());
    }
    std::set<uint64_t> landmarksConsidered;
    std::set_intersection(referenceLandmarks.begin(), referenceLandmarks.end(),
                          otherLandmarks.begin(), otherLandmarks.end(),
                          std::inserter(landmarksConsidered,landmarksConsidered.begin()));

    for(auto observation : referenceObservations) {
      Landmark & landmark = landmarks_.at(observation.second.landmarkId);

      if(landmarksConsidered.count(observation.second.landmarkId.value())==0) {
        continue;
      }
      twoPoseLink.errorTerm->addObservation(
            observation.first, observation.second.errorTerm, cauchyLossFunctionPtr_.get(),
            referenceState.pose, landmark.hPoint,
            referenceState.extrinsics.at(observation.first.cameraIndex), keepReferenceState,
            1.0);
    }
    for(auto observation : otherObservations) {
      Landmark & landmark = landmarks_.at(observation.second.landmarkId);

      if(landmarksConsidered.count(observation.second.landmarkId.value())==0) {
        continue;
      }
      twoPoseLink.errorTerm->addObservation(
            observation.first, observation.second.errorTerm, cauchyLossFunctionPtr_.get(),
            otherState.pose, landmark.hPoint,
            otherState.extrinsics.at(observation.first.cameraIndex), keepOtherState, 1.0);
    }

    if(landmarksConsidered.size()>0) {
      twoPoseLink.errorTerm->compute();

      numEdges.at(referenceStateId.value())--;
      numEdges.at(otherStateId.value())--;

      poseGraphEdges.push_back(
        PoseGraphEdge{twoPoseLink.errorTerm, referenceStateId, otherStateId});
    } else {

      numEdges.at(referenceStateId.value())--;
      numEdges.at(otherStateId.value())--;
    }
  }

  return true;
}

bool ViGraphEstimator::addExternalTwoPoseLink(
    const std::shared_ptr<ceres::TwoPoseGraphErrorConst> & twoPoseError,
    StateId referenceId, StateId otherId) {

  TwoPoseConstLink twoPoseConstLink;
  twoPoseConstLink.errorTerm = twoPoseError;
  State & referenceState = states_.at(referenceId);
  State & otherState = states_.at(otherId);
  twoPoseConstLink.state0 = referenceId;
  twoPoseConstLink.state1 = otherId;

  OKVIS_ASSERT_TRUE_DBG(
        Exception, !referenceState.twoPoseConstLinks.count(otherId)
        && !otherState.twoPoseConstLinks.count(referenceId),
        "duplicate two pose error, ID "<<otherId.value() << "<->" << referenceId.value())

  if (cameraParametersVec_.at(0).online_calibration.do_extrinsics) {
    std::vector<double *> pars;
    pars.push_back(referenceState.pose->parameters());
    pars.push_back(otherState.pose->parameters());
    for (const auto &extrinsics : referenceState.extrinsics) {
      pars.push_back(extrinsics->parameters());
    }
    twoPoseConstLink.residualBlockId
      = problem_->AddResidualBlock(
        std::static_pointer_cast<ceres::TwoPoseExtrinsicsGraphErrorConst>(
          twoPoseConstLink.errorTerm).get(), nullptr, pars);
  } else {
    twoPoseConstLink.residualBlockId
      = problem_->AddResidualBlock(std::static_pointer_cast<ceres::TwoPoseStandardGraphErrorConst>(
          twoPoseConstLink.errorTerm).get(), nullptr, referenceState.pose->parameters(),
          otherState.pose->parameters());
  }

  referenceState.twoPoseConstLinks[otherId] = twoPoseConstLink;
  otherState.twoPoseConstLinks[referenceId] = twoPoseConstLink;
  return true;
}

bool ViGraphEstimator::removeTwoPoseConstLinks(StateId convertStateId) {

  const std::map<StateId, TwoPoseConstLink> & poseGraphErrorTerms =
      states_.at(convertStateId).twoPoseConstLinks;
  for(auto iter=poseGraphErrorTerms.begin(); iter!=poseGraphErrorTerms.end(); ++iter) {

    StateId stateId = iter->first;

    OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(stateId).twoPoseConstLinks.count(convertStateId),
                          "no two pose const links present at ID " << convertStateId.value())
    states_.at(stateId).twoPoseConstLinks.erase(convertStateId);

    problem_->RemoveResidualBlock(iter->second.residualBlockId);
  }

  states_.at(convertStateId).twoPoseConstLinks.clear();

  return true;
}

bool ViGraphEstimator::removeTwoPoseConstLink(StateId pose_i, StateId pose_j)
{

  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(pose_i).twoPoseConstLinks.count(pose_j),
                        "two pose link does not exist")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(pose_j).twoPoseConstLinks.count(pose_i),
                        "two pose link does not exist")
  states_.at(pose_i).twoPoseConstLinks.erase(pose_j);
  problem_->RemoveResidualBlock(states_.at(pose_j).twoPoseConstLinks.at(pose_i).residualBlockId);
  states_.at(pose_j).twoPoseConstLinks.erase(pose_i);
  return true;
}

int ViGraphEstimator::convertToObservations(
    StateId convertStateId, std::set<LandmarkId> *createdLandmarks,
    std::set<StateId> *connectedStates,
    std::vector<ceres::TwoPoseGraphError::Observation> *allObservations)
{

  std::map<StateId, TwoPoseLink> & poseGraphErrorTerms =
      states_.at(convertStateId).twoPoseLinks;
  int ctr = 0;
  for(auto iter=poseGraphErrorTerms.begin(); iter!=poseGraphErrorTerms.end(); ++iter) {

    std::vector<StateId> stateIds;
    stateIds.push_back(iter->second.errorTerm->referencePoseId());
    stateIds.push_back(iter->second.errorTerm->otherPoseId());

    if(iter->second.errorTerm->referencePoseId() != convertStateId) {
      StateId stateId = iter->second.errorTerm->referencePoseId();
      if(connectedStates) {
        connectedStates->insert(stateId);  
      }
      states_.at(stateId).twoPoseLinks.erase(convertStateId);
    }
    if(iter->second.errorTerm->otherPoseId() != convertStateId) {
      StateId stateId = iter->second.errorTerm->otherPoseId();
      if(connectedStates) {
        connectedStates->insert(stateId);  
      }
      states_.at(stateId).twoPoseLinks.erase(convertStateId);
    }

    std::vector<ceres::TwoPoseGraphError::Observation> observations;
    std::vector<KeypointIdentifier> duplicates;
    iter->second.errorTerm->convertToReprojectionErrors(observations, duplicates);

    for(size_t i=0; i<observations.size(); ++i) {
      const ceres::TwoPoseGraphError::Observation & obs = observations.at(i);

      if(observations_.count(obs.keypointIdentifier)) {

        continue;
      }

      LandmarkId landmarkId(obs.hPoint->id());
      if(landmarks_.count(landmarkId) == 0) {
        addLandmark(landmarkId, obs.hPoint->estimate(), obs.hPoint->initialized());
      }

      if(createdLandmarks) {
        createdLandmarks->insert(landmarkId);
      }

      const bool useCauchy = obs.lossFunction != nullptr;
      addExternalObservation(obs.reprojectionError, landmarkId,
                             obs.keypointIdentifier, useCauchy);
      if(allObservations) {
        allObservations->push_back(observations.at(i));
      }

      ctr++;
    }

    problem_->RemoveResidualBlock(iter->second.residualBlockId);
  }

  poseGraphErrorTerms.clear();

  return ctr;
}

bool ViGraphEstimator::removeSpeedAndBiasPrior(StateId stateId)
{
  if(states_.at(stateId).speedAndBiasPrior.errorTerm) {
    problem_->RemoveResidualBlock(states_.at(stateId).speedAndBiasPrior.residualBlockId);
    states_.at(stateId).speedAndBiasPrior.errorTerm.reset();
    return true;
  }
  return false;
}

void ViGraphEstimator::clear()
{

  problem_.reset();

  mstFrameIds_.clear(); 
  mstGraphIdxs_.clear(); 
  mstEdges_.clear(); 
  states_.clear(); 
  landmarks_.clear(); 
  observations_.clear(); 

  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_shared<::ceres::Problem>(problemOptions);

  coObservationCounts_.clear(); 

  covisibilitiesComputed_ = true;
  visibleFrames_.clear(); 

  anyState_.clear(); 
}

int ViGraphEstimator::buildMst(const std::set<StateId> &states, bool keyframesOnly)
{

  const bool success0 = computeCovisibilities();
  OKVIS_ASSERT_TRUE(Exception, success0, "covisibilities could not be computed")

  mstFrameIds_.clear();
  mstGraphIdxs_.clear();
  mstEdges_.clear();

  int ctr = 0;
  for(auto val : states) {
    mstFrameIds_.push_back(val.value());
    mstGraphIdxs_[val.value()] = size_t(ctr);
    ++ctr;
  }

  int n2 = 0;
  for(auto i0 = coObservationCounts_.begin(); i0!=coObservationCounts_.end(); ++i0) {
    for(auto i1 = i0->second.begin(); i1!=i0->second.end(); ++i1) {
      bool considerFrame;
      if(keyframesOnly) {
        considerFrame = isKeyframe(StateId(i0->first)) && isKeyframe(StateId(i1->first));
      } else {
        considerFrame = true;
      }
      if(considerFrame && mstGraphIdxs_.count(i0->first) && mstGraphIdxs_.count(i1->first)) {
        n2++;
      }
    }
  }
  MstGraph graph(int(mstGraphIdxs_.size()), n2);
  for(auto i0 = coObservationCounts_.begin(); i0!=coObservationCounts_.end(); ++i0) {
    for(auto i1 = i0->second.begin(); i1!=i0->second.end(); ++i1) {
      bool considerFrame;
      if(keyframesOnly) {
        considerFrame = isKeyframe(StateId(i0->first)) && isKeyframe(StateId(i1->first));
      } else {
        considerFrame = true;
      }
      if(considerFrame && mstGraphIdxs_.count(i0->first) && mstGraphIdxs_.count(i1->first)) {
        const size_t id0 = mstGraphIdxs_.at(i0->first);
        const size_t id1 =mstGraphIdxs_.at(i1->first);
        int weight = -i1->second;
        graph.addEdge(int(id0), int(id1), weight);
      }
    }
  }

  const bool success = graph.kruskalMst(mstEdges_)>0;
  return success;
}

void ViGraphEstimator::checkSynchedStates(const ViGraphEstimator & other) const
{
  for(auto iter = other.states_.begin(); iter != other.states_.end(); ++iter) {
    OKVIS_ASSERT_TRUE(Exception, states_.count(iter->first) != 0, "inconsistent states")
    const State & otherState = iter->second;
    const State & thisState = states_.at(iter->first);
    OKVIS_ASSERT_TRUE(Exception, thisState.isKeyframe == otherState.isKeyframe,
                      "not same keyframe info")
    OKVIS_ASSERT_TRUE(
          Exception, thisState.pose->estimate().T().isApprox(otherState.pose->estimate().T()),
          "pose difference "<<iter->first.value()<<", Delta_r=" <<
          (thisState.pose->estimate().r()-otherState.pose->estimate().r()).transpose()
           << ", Delta_q=" <<
          (thisState.pose->estimate().q()*otherState.pose->estimate().q().inverse())
          .coeffs().transpose())
    OKVIS_ASSERT_TRUE(
          Exception,
          thisState.speedAndBias->estimate().isApprox(otherState.speedAndBias->estimate()),
          "speed and bias difference")
  }
}

bool ViGraphEstimator::isObservationlessSynched(const ViGraphEstimator & other) const
{
  for(auto iter = other.states_.begin(); iter != other.states_.end(); ++iter) {
    if(states_.count(iter->first) == 0) {
      std::cout << "inconsistent states" << std::endl;
      return false;
    }
    const State & otherState = iter->second;
    const State & thisState = states_.at(iter->first);
    if(thisState.isKeyframe != otherState.isKeyframe) {
      std::cout << "not same keyframe info" << std::endl;
      return false;
    }
    if(thisState.twoPoseLinks.size() != otherState.twoPoseLinks.size()) {
      if(thisState.twoPoseConstLinks.size() != otherState.twoPoseLinks.size()
         && thisState.twoPoseLinks.size() != otherState.twoPoseConstLinks.size()) {
        std::cout << "inconsistent two pose links ID=" << iter->first.value() << std::endl;
        return false;
      }
    }
    if(thisState.relativePoseLinks.size() != otherState.relativePoseLinks.size()) {
      std::cout << "inconsistent relative pose links ID=" << iter->first.value() << std::endl;
      return false;
    }

    if(!thisState.pose->estimate().T().isApprox(otherState.pose->estimate().T())) {
      std::cout << "pose difference "<<iter->first.value()<<", Delta_r="
                << (thisState.pose->estimate().r()-otherState.pose->estimate().r()).transpose()
                << ", Delta_q="
                << (thisState.pose->estimate().q()*otherState.pose->estimate().q().inverse())
                   .coeffs().transpose() << std::endl;
      return false;
    }
    if(!thisState.speedAndBias->estimate().isApprox(otherState.speedAndBias->estimate())) {
      std::cout << "speed and bias difference" << std::endl;
      return false;
    }
  }
  return true;
}

bool ViGraphEstimator::isSynched(const ViGraphEstimator & other) const
{

  if(observations_.size() != other.observations_.size()) {
    std::cout << "different number of observations" << std::endl;
    return false;
  }
  for(const auto& obs : observations_) {
    if(!other.observations_.count(obs.first)) {
      std::cout << "inconsistent observations 1 " << observations_.size()
                << " "<< other.observations_.size() << std::endl;
      return false;
    }
  }
  for(const auto& obs : other.observations_) {
    if(!observations_.count(obs.first)) {
      std::cout << "inconsistent observations 2" << observations_.size()
                << " "<< other.observations_.size() << std::endl;
      return false;
    }
    if(!observations_.at(obs.first).errorTerm->information().isApprox(
         obs.second.errorTerm->information())) {
      std::cout << "inconsistent observation information \n"
                << observations_.at(obs.first).errorTerm->information() << "\n"
                << obs.second.errorTerm->information() << std::endl;
      return false;
    }
  }

  int noErrorTerms = 0;

  for(const auto & lm : landmarks_) {
    if(!other.landmarks_.count(lm.first)) {
      std::cout << "inconsistent landmarks" << std::endl;
      return false;
    }
    for(const auto &obs : other.landmarks_.at(lm.first).observations) {
      if(!lm.second.observations.count(obs.first)) {
        std::cout << "inconsistent observations 3" << std::endl;
        return false;
      }

      Eigen::Vector2d err;
      const double* pars[3];
      pars[0] = other.states_.at(StateId(obs.first.frameId)).pose->parameters();
      pars[1] = other.landmarks_.at(lm.first).hPoint->parameters();
      pars[2] = other.states_.at(StateId(obs.first.frameId)).extrinsics.at(
            obs.first.cameraIndex)->parameters();
      obs.second.errorTerm->Evaluate(pars, err.data(), nullptr);
      const double r0 = err.transpose()*err;
      pars[0] = states_.at(StateId(obs.first.frameId)).pose->parameters();
      pars[1] = landmarks_.at(lm.first).hPoint->parameters();
      pars[2] = states_.at(StateId(obs.first.frameId)).extrinsics.at(
            obs.first.cameraIndex)->parameters();
      lm.second.observations.at(obs.first).errorTerm->Evaluate(pars, err.data(), nullptr);
      const double r1 = err.transpose()*err;
      noErrorTerms++;
      if(fabs(r0-r1)>1.0e-10) {
        std::cout << "inconsistent observations: deviating reprojection error" << std::endl;
        return false;
      }
    }
  }
  for(const auto& lm : other.landmarks_) {
    if(!landmarks_.count(lm.first)) {
      std::cout << "inconsistent landmarks" << std::endl;
      return false;
    }
    if(!landmarks_.at(lm.first).hPoint->estimate().isApprox(lm.second.hPoint->estimate())) {
      std::cout << "landmark difference " << lm.first.value() << std::endl;
      return false;
    }
    for(const auto& obs : landmarks_.at(lm.first).observations) {
      if(!lm.second.observations.count(obs.first)) {
        std::cout << lm.first.value() << " " << lm.second.observations.size()
                  << " " << landmarks_.at(lm.first).observations.size() << std::endl;
        std::cout << "inconsistent observations 4" << std::endl;
        return false;
      }

      Eigen::Vector2d err;
      const double* pars[3];
      pars[0] = other.states_.at(StateId(obs.first.frameId)).pose->parameters();
      pars[1] = other.landmarks_.at(lm.first).hPoint->parameters();
      pars[2] = other.states_.at(StateId(obs.first.frameId)).extrinsics.at(
            obs.first.cameraIndex)->parameters();
      lm.second.observations.at(obs.first).errorTerm->Evaluate(pars, err.data(), nullptr);
      const double r0 = err.transpose()*err;
      pars[0] = states_.at(StateId(obs.first.frameId)).pose->parameters();
      pars[1] = landmarks_.at(lm.first).hPoint->parameters();
      pars[2] = states_.at(StateId(obs.first.frameId)).extrinsics.at(
            obs.first.cameraIndex)->parameters();
      obs.second.errorTerm->Evaluate(pars, err.data(), nullptr);
      const double r1 = err.transpose()*err;
      if(fabs(r0-r1)>1.0e-10) {
        std::cout << "inconsistent observations: deviating reprojection error" << std::endl;
        return false;
      }
    }
  }
  int noTwoPoseLinks = 0;
  for(auto iter = other.states_.begin(); iter != other.states_.end(); ++iter) {
    if(states_.count(iter->first) == 0) {
      std::cout << "inconsistent states" << std::endl;
      return false;
    }
    const State & otherState = iter->second;
    const State & thisState = states_.at(iter->first);
    if(thisState.isKeyframe != otherState.isKeyframe) {
      std::cout << "not same keyframe info" << std::endl;
      return false;
    }
    if(thisState.twoPoseLinks.size() != otherState.twoPoseLinks.size()) {
      if(thisState.twoPoseConstLinks.size() != otherState.twoPoseLinks.size()
         && thisState.twoPoseLinks.size() != otherState.twoPoseConstLinks.size()) {
        std::cout << "inconsistent two pose links ID=" << iter->first.value() << std::endl;
        return false;
      }

      std::vector<ceres::ErrorInterface*> thisErrors;
      std::vector<StateId> thisP0, thisP1, otherP0, otherP1;
      std::vector<ceres::ErrorInterface*> otherErrors;
      for(const auto& thisLink : thisState.twoPoseLinks) {
        thisErrors.push_back(thisLink.second.errorTerm.get());
        thisP0.push_back(thisLink.second.state0);
        thisP1.push_back(thisLink.second.state1);
      }
      for(const auto& thisLink : thisState.twoPoseConstLinks) {
        thisErrors.push_back(thisLink.second.errorTerm.get());
        thisP0.push_back(thisLink.second.state0);
        thisP1.push_back(thisLink.second.state1);
      }
      for(const auto& otherLink : otherState.twoPoseLinks) {
        otherErrors.push_back(otherLink.second.errorTerm.get());
        otherP0.push_back(otherLink.second.state0);
        otherP1.push_back(otherLink.second.state1);
      }
      for(const auto& otherLink : otherState.twoPoseConstLinks) {
        otherErrors.push_back(otherLink.second.errorTerm.get());
        otherP0.push_back(otherLink.second.state0);
        otherP1.push_back(otherLink.second.state1);
      }
      for(size_t i = 0; i<thisErrors.size(); ++i) {
        const double* pars[2];
        pars[0]=states_.at(thisP0[i]).pose->parameters();
        pars[1]=states_.at(thisP1[i]).pose->parameters();
        Eigen::Matrix<double,6,1> err;
        thisErrors[i]->EvaluateWithMinimalJacobians(pars, err.data(), nullptr, nullptr);
        const double r0 = err.transpose()*err;
        pars[0]=other.states_.at(otherP0[i]).pose->parameters();
        pars[1]=other.states_.at(otherP1[i]).pose->parameters();
        otherErrors[i]->EvaluateWithMinimalJacobians(pars, err.data(), nullptr, nullptr);
        const double r1 = err.transpose()*err;
        noTwoPoseLinks++;
        if(fabs(r0-r1)>1.0e-10) {
          std::cout << "inconsistent two pose links: deviating error term" << std::endl;
          return false;
        }
      }
    }

    if(thisState.relativePoseLinks.size() != otherState.relativePoseLinks.size()) {
      std::cout << "inconsistent relative pose links ID=" << iter->first.value() << std::endl;
      return false;
    }

    if(!thisState.pose->estimate().T().isApprox(otherState.pose->estimate().T())) {
      std::cout << "pose difference "<<iter->first.value()<<", Delta_r="
                << (thisState.pose->estimate().r()-otherState.pose->estimate().r()).transpose()
                << ", Delta_q="
                << (thisState.pose->estimate().q()*otherState.pose->estimate().q().inverse())
                   .coeffs().transpose() << std::endl;
      return false;
    }
    if(!thisState.speedAndBias->estimate().isApprox(otherState.speedAndBias->estimate())) {
      std::cout << "speed and bias difference" << std::endl;
      return false;
    }

    if(thisState.previousImuLink.errorTerm) {
      const double* pars[4];
      auto iterTmp = iter;
      iterTmp--;
      pars[0] = iterTmp->second.pose->parameters();
      pars[1] = iterTmp->second.speedAndBias->parameters();
      pars[2] = iter->second.pose->parameters();
      pars[3] = iter->second.speedAndBias->parameters();
      Eigen::Matrix<double,15,1> err;
      thisState.previousImuLink.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r0 = err.transpose()*err;
      otherState.previousImuLink.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r1 = err.transpose()*err;
      noErrorTerms++;
      if(fabs(r0-r1)>1.0e-10) {
        std::cout << "inconsistent imu error: " << otherState.previousImuLink.residualBlockId
                  << " deviating " << r0-r1 << std::endl;
        return false;
      }
    }

    if(thisState.posePrior.errorTerm) {
      const double* pars[1];
      pars[0] = iter->second.pose->parameters();
      Eigen::Matrix<double,6,1> err;
      thisState.posePrior.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r0 = err.transpose()*err;
      otherState.posePrior.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r1 = err.transpose()*err;
      noErrorTerms++;
      if(fabs(r0-r1)>1.0e-10) {
        std::cout << "inconsistent pose prior: deviating error term" << std::endl;
        return false;
      }
    }
    if(thisState.speedAndBiasPrior.errorTerm) {
      const double* pars[1];
      pars[0] = iter->second.speedAndBias->parameters();
      Eigen::Matrix<double,9,1> err;
      thisState.speedAndBiasPrior.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r0 = err.transpose()*err;
      otherState.speedAndBiasPrior.errorTerm->Evaluate(pars,err.data(),nullptr);
      const double r1 = err.transpose()*err;
      noErrorTerms++;
      if(fabs(r0-r1)>1.0e-10) {
        std::cout << "inconsistent speed and bias prior: deviating error term" << std::endl;
        return false;
      }
    }
  }
  noErrorTerms += noTwoPoseLinks/2; 

  if(problem_->NumResidualBlocks() != other.problem_->NumResidualBlocks()){
    std::cout << "unaccounted residual blocks" << std::endl;
    return false;
  }
  if(problem_->NumResidualBlocks() != noErrorTerms){
    std::cout << "unaccounted residual blocks "<< problem_->NumResidualBlocks()
              << " vs " << noErrorTerms << std::endl;
    return false;
  }

  return true;
}

}  
