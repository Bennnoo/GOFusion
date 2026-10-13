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
 * @file ViSlamBackend.cpp
 * @brief Source file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <okvis/ViSlamBackend.hpp>
#include <okvis/PseudoInverse.hpp>
#include <okvis/Component.hpp>
#include <okvis/timing/Timer.hpp>

namespace okvis {

namespace {

class FinalBaIterationCallback final : public ::ceres::IterationCallback {
 public:
  FinalBaIterationCallback(
      ViSlamBackend::FinalBaProgressCallback callback,
      const int stage,
      const int max_iterations)
      : callback_(std::move(callback)),
        stage_(stage),
        max_iterations_(max_iterations)
  {
  }

  ::ceres::CallbackReturnType operator()(
      const ::ceres::IterationSummary& summary) override
  {
    callback_(stage_,
              std::min(summary.iteration + 1, max_iterations_),
              max_iterations_);
    return ::ceres::SOLVER_CONTINUE;
  }

 private:
  ViSlamBackend::FinalBaProgressCallback callback_;
  int stage_ = 0;
  int max_iterations_ = 0;
};

class ScopedSolverCallback final {
 public:
  ScopedSolverCallback(::ceres::Solver::Options& options,
                       ::ceres::IterationCallback* callback)
      : options_(options), callback_(callback)
  {
    if (callback_) options_.callbacks.push_back(callback_);
  }

  ~ScopedSolverCallback()
  {
    if (!callback_) return;
    auto& callbacks = options_.callbacks;
    const auto it = std::find(callbacks.begin(), callbacks.end(), callback_);
    if (it != callbacks.end()) callbacks.erase(it);
  }

 private:
  ::ceres::Solver::Options& options_;
  ::ceres::IterationCallback* callback_ = nullptr;
};

}  

static const double minDeltaT = 2.0; 
static int numRealtimePoseGraphFrames = 12;
static const int numPoseGraphFrames = 12;

int ViSlamBackend::addCamera(const CameraParameters &cameraParameters)
{
  fullGraph_.addCamera(cameraParameters);
  return realtimeGraph_.addCamera(cameraParameters);
}

int ViSlamBackend::addImu(const ImuParameters &imuParameters)
{
  fullGraph_.addImu(imuParameters);
  return realtimeGraph_.addImu(imuParameters);
}

int ViSlamBackend::addGps(const GpsParameters &gpsParameters)
{
  fullGraph_.addGps(gpsParameters);
  return realtimeGraph_.addGps(gpsParameters);
}

bool ViSlamBackend::addGpsMeasurementsOnAllGraphs(GpsMeasurementDeque& inputgpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque){
  if(realtimeGraph_.gpsParametersVec_.empty()) {
    return false;
  }

  GpsMeasurementDeque gpsMeasurementDeque;
  if(realtimeGraph_.gpsParametersVec_.back().robustGpsInit){
    [[maybe_unused]] int valid = realtimeGraph_.checkValidGpsMeasurements(inputgpsMeasurementDeque, gpsMeasurementDeque);
  }
  else {
    gpsMeasurementDeque = inputgpsMeasurementDeque;
  }

  if(gpsMeasurementDeque.size() > 0){

    bool needsGpsReInit = realtimeGraph_.needsGpsReInit(); 
    if(needsGpsReInit) {
      realtimeGraph_.reInitGpsExtrinsics();
    }

    fullGraph_.setGpsStatus(realtimeGraph_.getGpsStatus());

    if(!isLoopClosing_ && !isLoopClosureAvailable_){ 

      realtimeGraph_.addGpsMeasurements(gpsMeasurementDeque,imuMeasurementDeque,nullptr);
      fullGraph_.addGpsMeasurements(gpsMeasurementDeque, imuMeasurementDeque,nullptr);
    }
    else{ 

      std::deque<StateId> sids; 
      realtimeGraph_.addGpsMeasurements(gpsMeasurementDeque,imuMeasurementDeque,&sids);

      for(size_t i = 0; i < gpsMeasurementDeque.size(); i++){
        addGpsBacklog_.push_back(AddGpsBacklog{sids.at(i), gpsMeasurementDeque.at(i), imuMeasurementDeque, needsGpsReInit});
      }
    }

    kinematics::Transformation T_GW_init;
    if(realtimeGraph_.initializationStrategy(T_GW_init)){
      realtimeGraph_.addGpsInitFactors();
      fullGraph_.addGpsInitFactors();
      realtimeGraph_.setGpsExtrinsics(T_GW_init);
      fullGraph_.setGpsExtrinsics(T_GW_init);
    }

    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      [[maybe_unused]] bool requires_gps_alignment = tryGpsAlignment();
    }

    if(!gpsObservability_){
      gpsObservability_ = realtimeGraph_.isGpsObservable();
    }
    return true; 
  }
  else {
    return false; 
  }
}

bool ViSlamBackend::tryGpsAlignment(){
  Eigen::Vector3d posAlignVec;
  StateId gpsDropId;
  StateId alignId;
  okvis::kinematics::Transformation T_GW_new;

  bool needInitialAlign = realtimeGraph_.needsInitialGpsAlignment();
  if(needInitialAlign){
    fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
    addGpsAlignmentFrame(StateId(1));
    realtimeGraph_.resetInitialGpsAlignment();
    fullGraph_.resetInitialGpsAlignment();
    return true;
  }

  [[maybe_unused]] bool _ = fullGraph_.needsFullGpsAlignment(gpsDropId, alignId, T_GW_new);
  bool needFullAlign = realtimeGraph_.needsFullGpsAlignment(gpsDropId, alignId, T_GW_new);

  if(needFullAlign){

    attemptFullGpsAlignment(gpsDropId,alignId, T_GW_new);

    std::vector<StateId> updatedStatesRealtime;
    optimiseRealtimeGraph(50, updatedStatesRealtime);

    addGpsAlignmentFrame(gpsDropId);
    realtimeGraph_.resetFullGpsAlignment();
    fullGraph_.resetFullGpsAlignment();
    return true;
  }
  else{

    bool needPosAlign = realtimeGraph_.needsPosGpsAlignment(gpsDropId, alignId, posAlignVec);
    if(needPosAlign){
      attemptPosGpsAlignment(gpsDropId,alignId, posAlignVec);
      addGpsAlignmentFrame(gpsDropId);
      realtimeGraph_.resetPosGpsAlignment();
      fullGraph_.resetPosGpsAlignment();
      return true;
    }
    else{
      return false;
    }
  }
}

bool ViSlamBackend::addStates(MultiFramePtr multiFrame, const ImuMeasurementDeque &imuMeasurements,
                              bool asKeyframe)
{
  AuxiliaryState auxiliaryState;
  auxiliaryState.isImuFrame = true;
  auxiliaryState.isKeyframe = asKeyframe;
  if(multiFrames_.empty()) {

    OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "not allowed")
    const StateId id = realtimeGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                                          multiFrame->cameraSystem());
    multiFrame->setId(id.value());
    fullGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                              multiFrame->cameraSystem());
    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; 
    imuFrames_.insert(id);
    return (id.value()==1 && id.isInitialised());
  } else {
    const StateId id = realtimeGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements,
                                                         asKeyframe);

    for (size_t i = 0; i < multiFrame->numFrames(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        const kinematics::Transformation T_SCi = extrinsics(id, i);
        multiFrame->setExtrinsics(i, T_SCi);
      }
    }

    multiFrame->setId(id.value());

    if(isLoopClosing_ || isLoopClosureAvailable_) {
      addStatesBacklog_.push_back(AddStatesBacklog{multiFrame->timestamp(), id, imuMeasurements});
      touchedStates_.insert(id);
    } else {

      fullGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements, asKeyframe);
    }

    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; 
    imuFrames_.insert(id);
    return id.isInitialised();
  }
}

void ViSlamBackend::printStates(StateId stateId, std::ostream &buffer) const
{
  const ViGraph::State & state = realtimeGraph_.states_.at(stateId);
  if(state.isKeyframe) {
    buffer << "KF ";
  }
  buffer << "pose: ";
  if (state.pose->fixed()) buffer << "(";
  buffer << "id=" << stateId.value() << ":";
  if (state.pose->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "speedAndBias: ";
  if (state.speedAndBias->fixed()) buffer << "(";
  buffer << "id=" << state.speedAndBias->id() << ":";
  if (state.speedAndBias->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "extrinsics: ";
  for (size_t i = 0; i < state.extrinsics.size(); ++i) {
    uint64_t id = state.extrinsics.at(i)->id();
    if (state.extrinsics.at(i)->fixed()) buffer << "(";
    buffer << "id=" << id << ":";
    if (state.extrinsics.at(i)->fixed()) buffer << ")";
    if (i+1 < state.extrinsics.size()) buffer << ", ";
  }
  buffer << std::endl;
}

bool ViSlamBackend::getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const
{
  return realtimeGraph_.getLandmark(landmarkId, mapPoint);
}

size_t ViSlamBackend::getLandmarks(MapPoints & landmarks) const
{
  return realtimeGraph_.getLandmarks(landmarks);
}

double ViSlamBackend::trackingQuality(StateId id) const
{
  const MultiFramePtr frame = multiFrame(id);
  const size_t numFrames = frame->numFrames();
  std::set<LandmarkId> landmarks;
  std::vector<cv::Mat> matchesImg(numFrames);

  int intersectionCount = 0;
  int unionCount = 0;
  int matchedPoints = 0;
  for (size_t im = 0; im < numFrames; ++im) {
    const int rows = frame->image(im).rows/10;
    const int cols = frame->image(im).cols/10;
    const double radius = double(std::min(rows,cols))*kptradius_;
    matchesImg.at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
    const size_t num = frame->numKeypoints(im);
    cv::KeyPoint keypoint;
    for (size_t k = 0; k < num; ++k) {
      frame->getCvKeypoint(im, k, keypoint);
      uint64_t lmId = frame->landmarkId(im, k);
      if (lmId != 0 && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {

        for(const auto & obs : realtimeGraph_.landmarks_.at(LandmarkId(lmId)).observations) {
          if(obs.first.frameId != id.value()) {
            matchedPoints++;
            cv::circle(matchesImg.at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                       cv::FILLED);
            break;
          }
        }
      }
    }

    const int pointArea = int(radius*radius*M_PI);
    intersectionCount += std::max(0,cv::countNonZero(matchesImg.at(im)) - pointArea);
    unionCount += rows*cols - pointArea;
  }
  return matchedPoints < 8 ? 0.0 : double(intersectionCount)/double(unionCount);
}

bool ViSlamBackend::setKeyframe(StateId id, bool isKeyframe) {
  auxiliaryStates_.at(id).isKeyframe = isKeyframe;
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    fullGraph_.setKeyframe(id, isKeyframe);
  }
  return realtimeGraph_.setKeyframe(id, isKeyframe);
}

bool ViSlamBackend::addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &landmark,
                                bool isInitialised)
{
  bool success = realtimeGraph_.addLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    success &= fullGraph_.addLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}

LandmarkId ViSlamBackend::addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  const LandmarkId lmId = realtimeGraph_.addLandmark(homogeneousPoint, initialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(lmId);
  } else {
    fullGraph_.addLandmark(lmId, homogeneousPoint, initialised);
  }
  return lmId;
}

bool ViSlamBackend::setLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark,
                                bool isInitialised) {
  bool success = realtimeGraph_.setLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    fullGraph_.setLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}

bool ViSlamBackend::setObservationInformation(
    StateId stateId, size_t camIdx, size_t keypointIdx, const Eigen::Matrix2d & information) {
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.observations_.at(kid).errorTerm->setInformation(information);
  }
  realtimeGraph_.observations_.at(kid).errorTerm->setInformation(information);
  return true;
}

bool ViSlamBackend::removeObservation(StateId stateId, size_t camIdx,
                                      size_t keypointIdx)
{
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.removeObservation(kid);
  }
  bool success = realtimeGraph_.removeObservation(kid);
  multiFrames_.at(stateId)->setLandmarkId(camIdx, keypointIdx, 0);
  return success;
}

bool ViSlamBackend::convertToPoseGraphMst(const std::set<StateId> & framesToConvert,
                                          const std::set<StateId> & framesToConsider,
                                          std::set<StateId> & affectedFrames) {

  for(auto pose : framesToConvert) {
    ViGraph::State& state = realtimeGraph_.states_.at(pose);
    kinematics::Transformation T_SW = state.pose->estimate().inverse();
    MultiFramePtr mFrame = multiFrames_.at(pose);
    for(size_t i=0; i<mFrame->numFrames(); ++i) {
      for(size_t k=0; k<mFrame->numKeypoints(i); ++k) {
        uint64_t lmId = mFrame->landmarkId(i,k);
        if(lmId && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {
          Eigen::Vector4d landmark = realtimeGraph_.landmark(LandmarkId(lmId));
          mFrame->setLandmark(i, k, T_SW*landmark,
                              realtimeGraph_.isLandmarkInitialised(LandmarkId(lmId)));
        }
      }
    }
  }

  std::vector<ViGraphEstimator::PoseGraphEdge> poseGraphEdges;
  std::vector<std::pair<StateId,StateId>> removedTwoPoseErrors;
  std::vector<KeypointIdentifier> removedObservations;
  realtimeGraph_.convertToPoseGraphMst(
        framesToConvert, framesToConsider, &poseGraphEdges, &removedTwoPoseErrors,
        &removedObservations);

  for (auto addedEdge : poseGraphEdges) {
    affectedFrames.insert(addedEdge.otherId);
    affectedFrames.insert(addedEdge.referenceId);
  }
  for (auto removedEdge : removedTwoPoseErrors) {
    affectedFrames.insert(removedEdge.first);
    affectedFrames.insert(removedEdge.second);
  }

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & poseGraphEdge : poseGraphEdges) {
      touchedStates_.insert(poseGraphEdge.referenceId);
      touchedStates_.insert(poseGraphEdge.otherId);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      touchedStates_.insert(removedTwoPoseError.first);
      touchedStates_.insert(removedTwoPoseError.second);
    }
    for(auto obs : removedObservations) {
      touchedStates_.insert(StateId(obs.frameId));
      if(fullGraph_.observations_.count(obs)) {
        LandmarkId lmId = fullGraph_.observations_.at(obs).landmarkId;
        if(lmId.isInitialised()) {
          touchedLandmarks_.insert(fullGraph_.observations_.at(obs).landmarkId); 
        }
      }
    }
  } else {

    for(auto obs : removedObservations) {
      fullGraph_.removeObservation(obs);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      fullGraph_.removeTwoPoseConstLink(removedTwoPoseError.first, removedTwoPoseError.second);
    }
    for(const auto & poseGraphEdge : poseGraphEdges) {
      fullGraph_.addExternalTwoPoseLink(
            poseGraphEdge.poseGraphError->cloneTwoPoseGraphErrorConst(),
            poseGraphEdge.referenceId, poseGraphEdge.otherId);

    }
  }

  return true;
}

int ViSlamBackend::expandKeyframe(StateId keyframe)
{
  OKVIS_ASSERT_TRUE(Exception, imuFrames_.count(keyframe) == 0, "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, keyFrames_.count(keyframe) || loopClosureFrames_.count(keyframe),
                    "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(keyframe).twoPoseLinks.size()>0,
                    "keyframe must have pose-graph links")
  int ctr = 0;
  std::set<LandmarkId> lms;
  std::set<StateId> cnncts;
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(keyframe, &lms, &cnncts, &allObservations);

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(cnncts.begin(), cnncts.end());
    touchedLandmarks_.insert(lms.begin(), lms.end());
  } else {
    fullGraph_.removeTwoPoseConstLinks(keyframe);
    for(auto lm : lms) {
      const auto & landmark = realtimeGraph_.landmarks_.at(lm);
      if(!fullGraph_.landmarkExists(lm)) {
        fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
        fullGraph_.setLandmarkQuality(lm, landmark.quality);
      }
    }
    for(const auto & obs : allObservations) {
      const bool useCauchy = obs.lossFunction != nullptr;
      LandmarkId landmarkId(obs.hPoint->id());
      fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                             obs.keypointIdentifier, useCauchy);
    }
  }

  for(auto cnnct : cnncts) {
    if(loopClosureFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      keyFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
    if(keyFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      loopClosureFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
  }

  return ctr;
}

void ViSlamBackend::eliminateImuFrames(size_t numImuFrames, std::set<StateId> & )
{
  std::set<StateId> imuFrames = imuFrames_; 
  for(auto id : imuFrames) {
    if(imuFrames_.size() <= numImuFrames) {
      break; 
    }
    if(realtimeGraph_.states_.at(id).isKeyframe) {

      imuFrames_.erase(id);
      keyFrames_.insert(id);
      components_.at(currentComponentIdx_).poseIds.insert(id);
      if(loopClosureFrames_.count(id)) {
        loopClosureFrames_.erase(id); 
      }
      auxiliaryStates_.at(id).isImuFrame = false;
    } else {
      StateId refId = mostOverlappedStateId(id, false);
      auto observations = realtimeGraph_.states_.at(id).observations;
      realtimeGraph_.removeAllObservations(id);
      realtimeGraph_.eliminateStateByImuMerge(id, refId);

      if(loopClosureFrames_.count(id)) {
        OKVIS_THROW(Exception, "bug")
      }

      if(isLoopClosing_ || isLoopClosureAvailable_) {
        for(const auto & obs : observations) {
          touchedLandmarks_.insert(obs.second.landmarkId);
        }
        eliminateStates_[id] = refId;
      } else {
        fullGraph_.removeAllObservations(id); 
        fullGraph_.eliminateStateByImuMerge(id, refId); 
      }
      imuFrames_.erase(id);
      multiFrames_.erase(id);
      auxiliaryStates_.erase(id);
    }
  }
}

bool ViSlamBackend::applyStrategy(size_t numKeyframes,
                                  size_t numLoopClosureFrames,
                                  size_t numImuFrames,
                                  std::set<StateId> &affectedFrames,
                                  bool expand)
{

  TimerSwitchable t1("7.1 check/handle lost");
  if(imuFrames_.size()>1 && !keyFrames_.empty()) {
    auto iter = auxiliaryStates_.rbegin();
    const double quality = trackingQuality(iter->first);
    if (quality < 0.01) {
      if (quality > 0.00001) {
        LOG(WARNING) << "Tracking quality weak: quality=" << quality;
      } else {
        LOG(WARNING) << "TRACKING FAILURE: quality=" << quality;
      }

    }
  }
  t1.stop();

  TimerSwitchable t2("7.2 eliminate IMU frames");
  eliminateImuFrames(numImuFrames, affectedFrames);
  t2.stop();

  StateId currentKeyframeId = currentKeyframeStateId();
  StateId currentFrameId = realtimeGraph_.currentStateId();

  if(!currentKeyframeId.isInitialised()) {
    return true; 
  }

  bool keyFrameEliminated = false;
  int ctrPg = 0;
  if(keyFrames_.size() > numKeyframes) {
    TimerSwitchable t3("7.3 convert to posegraph");
    while(keyFrames_.size() > numKeyframes) {

      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto keyFrame : keyFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, keyFrame),
                                   realtimeGraph_.covisibilities(currentKeyframeId, keyFrame));
        if(keyFrame == *keyFrames_.begin()){
          if(coObs>=2) {
            continue; 
          }
        }
        if(coObs < minObservations) {
          minObservations = coObs;
          minId = keyFrame;
        }
      }
      StateId maxId;
      int maxCoObs=0;
      std::set<StateId> framesToConsider;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {

          framesToConsider.insert(frameId);
        }
      }

      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; 
      keyFrames_.erase(minId);
      for(auto id : keyFrames_) {
        auxiliaryStates_.at(minId).recentLoopClosureFrames.insert(id); 
      }
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      keyFrameEliminated = true;
      if(maxCoObs == 0) {

        realtimeGraph_.removeAllObservations(minId);

        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if(convertToPosegraphFrames.size() > 0) {
        bool success = convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider,
                                             affectedFrames);
        ctrPg++;
        OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(minId).observations.size()==0,
                          "observations at ID " << minId.value() << " , success=" << int(success))
        if(ctrPg>=3) {
          break; 
        }
      }
    }
    t3.stop();
  }

  TimerSwitchable t4("7.4 freezing");
  if(keyFrameEliminated) {
    auto iter = auxiliaryStates_.rbegin();
    StateId oldestKeyFrameId;
    for (size_t p = 0; p<(numKeyframes+numImuFrames) && iter!=auxiliaryStates_.rend(); ++p) {
      iter++;
    }
    if(iter!=auxiliaryStates_.rend()) {
      oldestKeyFrameId = iter->first;
    }

    int ctr = 0;
    for(auto iter = realtimeGraph_.states_.find(oldestKeyFrameId);
        iter != realtimeGraph_.states_.end(); --iter) {
      if(ctr==numRealtimePoseGraphFrames) {
        Time freezeTime = realtimeGraph_.states_.rbegin()->second.timestamp;
        while((freezeTime - iter->second.timestamp).toSec() < minDeltaT) {
          if(iter==realtimeGraph_.states_.begin()) {
            break;
          }
          --iter;
        }
        if(iter!=realtimeGraph_.states_.begin()) {

          const StateId freezeId = std::max(lastFreeze_, iter->first);

          lastFreeze_ = freezeId;
          if(realtimeGraph_.states_.find(freezeId) != realtimeGraph_.states_.begin()) {
            realtimeGraph_.freezePosesUntil(freezeId);
          }

          realtimeGraph_.freezeSpeedAndBiasesUntil(freezeId);
        }
        break;
      }
      if(iter==realtimeGraph_.states_.begin()) {
        break;
      }
      ctr++;
    }
  }
  t4.stop();

  int ctrLc = 0;
  if(loopClosureFrames_.size() > 0) {
    TimerSwitchable t5("7.5 convert to posegraph loop-closure frames");
    do {

      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto loopClosureFrame : loopClosureFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, loopClosureFrame),
                                realtimeGraph_.covisibilities(currentKeyframeId, loopClosureFrame));

        if(coObs < minObservations) {
          minObservations = coObs;
          minId = loopClosureFrame;
        }
      }

      if(minObservations!=0 && loopClosureFrames_.size()<=numLoopClosureFrames) {
        break;
      }

      std::set<StateId> framesToConsider;
      StateId maxId;
      int maxCoObs = 0;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {

          framesToConsider.insert(frameId);
        }
      }

      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; 
      loopClosureFrames_.erase(minId);
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      if(maxCoObs == 0) {

        realtimeGraph_.removeAllObservations(minId);

        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if(convertToPosegraphFrames.size() > 0 && maxCoObs > 0) {
        convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider, affectedFrames);
        ++ctrLc;
        if(ctrLc>=3) {
          break; 
        }
      }
    } while (loopClosureFrames_.size() > numLoopClosureFrames);
    t5.stop();
  }

  if(expand && ctrLc<3 && ctrPg<3) {
    TimerSwitchable t6("7.6 expand");
    currentKeyframeId = currentKeyframeStateId();
    if(currentKeyframeId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentKeyframeId).twoPoseLinks.size()>0) {
        expandKeyframe(currentKeyframeId);
      }
    }
    const StateId currentLoopclosureFrameId = currentLoopclosureStateId();
    if(currentLoopclosureFrameId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentLoopclosureFrameId).twoPoseLinks.size()>0) {
        expandKeyframe(currentLoopclosureFrameId);
      }
    }
    t6.stop();
  }

  return true;
}

void ViSlamBackend::optimiseRealtimeGraph(
  int numIter, std::vector<StateId> & updatedStates, int numThreads, bool verbose,
  bool onlyNewestState, bool isInitialised)
{

  std::unique_ptr<ceres::PoseError> initialFixation;
  ::ceres::ResidualBlockId initialFixationId = nullptr;
  if(!isInitialised){
    Eigen::Matrix<double, 6, 1> informationDiag = Eigen::Matrix<double, 6, 1>::Ones();
    informationDiag[0] = 1.0e8;
    informationDiag[1] = 1.0e8;
    informationDiag[2] = 1.0e8;
    informationDiag[3] = 0.0;
    informationDiag[4] = 0.0;
    informationDiag[5] = 0.0;
    initialFixation.reset(
      new ceres::PoseError(realtimeGraph_.states_.at(StateId(1)).pose->estimate(), informationDiag));
    initialFixationId = realtimeGraph_.problem_->AddResidualBlock(
        initialFixation.get(), nullptr,
        realtimeGraph_.states_.rbegin()->second.pose->parameters());
    kinematics::Transformation T_WS(
        realtimeGraph_.states_.at(StateId(1)).pose->estimate().r(),
        realtimeGraph_.states_.rbegin()->second.pose->estimate().q());
    realtimeGraph_.setPose(realtimeGraph_.states_.rbegin()->first, T_WS);
  }

  bool frozen = false;
  StateId unfreezeId;
  if(onlyNewestState) {

    for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
        ++riter) {
      if(riter->second.pose->fixed()) {
        break;
      } else {
        unfreezeId = riter->first;
      }
    }

    auto riter = realtimeGraph_.states_.rbegin();
    riter++;
    if(riter != realtimeGraph_.states_.rend()) {
      realtimeGraph_.freezePosesUntil(riter->first);
      realtimeGraph_.freezeSpeedAndBiasesUntil(riter->first);
      frozen = true;
    }
    for(const auto & lm : realtimeGraph_.landmarks_) {
      realtimeGraph_.problem_->SetParameterBlockConstant(lm.second.hPoint->parameters());
    }

    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockConstant(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }
  }

  realtimeGraph_.options_.linear_solver_type = ::ceres::DENSE_SCHUR;
  realtimeGraph_.optimise(numIter, numThreads, verbose);

  if(onlyNewestState) {

    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockVariable(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }

    if(frozen) {
      realtimeGraph_.unfreezePosesFrom(unfreezeId);
      realtimeGraph_.unfreezeSpeedAndBiasesFrom(unfreezeId);
    }
    if(onlyNewestState) {
      for(const auto & lm : realtimeGraph_.landmarks_) {
        realtimeGraph_.problem_->SetParameterBlockVariable(lm.second.hPoint->parameters());
      }
    }

    if(initialFixation && initialFixationId) {
      realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
    }

    auto riter = realtimeGraph_.states_.rbegin();
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
    }

    return;
  }

  if(!onlyNewestState) {
    realtimeGraph_.updateLandmarks();
  }

  for (auto id : updatedStatesLoopClosureAttempt_) {
    updatedStates.push_back(id); 
  }
  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()) {
      break; 
    }
    if(!updatedStatesLoopClosureAttempt_.count(riter->first)) {
       updatedStates.push_back(riter->first);
    }

    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
      fullState.T_GW->setEstimate(riter->second.T_GW->estimate());
    }
  }

  if(!gpsObservability_){ 

      if(!isLoopClosing_ && !isLoopClosureAvailable_){
          fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
      }
  }
  else{ 
      if(!realtimeGraph_.isGpsFixed()){ 
          if(!isLoopClosing_ && !isLoopClosureAvailable_){ 
              realtimeGraph_.freezeGpsExtrinsics();
              fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
              fullGraph_.freezeGpsExtrinsics();
          }
      }
  }
  updatedStatesLoopClosureAttempt_.clear(); 

  if(!isLoopClosing_ && !isLoopClosureAvailable_) {
    for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
        ++iter) {
      fullGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                             iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(!riter->second.previousImuLink.errorTerm) {
      continue;
    }
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      if (fullGraph_.imuParametersVec_.at(0).use) {
        std::static_pointer_cast<ceres::ImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(
            *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            riter->second.previousImuLink.errorTerm));
      }
    }
    if(riter->second.pose->fixed()) {
      break;
    }
  }

  if(initialFixation && initialFixationId) {
    realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
  }

}

bool ViSlamBackend::setOptimisationTimeLimit(double timeLimit, int minIterations)
{

  return realtimeGraph_.setOptimisationTimeLimit(timeLimit, minIterations);
}

bool ViSlamBackend::isLandmarkAdded(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkAdded(landmarkId);
}

bool ViSlamBackend::isLandmarkInitialised(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkInitialised(landmarkId);
}

bool ViSlamBackend::setPose(StateId id, const kinematics::TransformationCacheless &pose)
{
  bool success = realtimeGraph_.setPose(id, pose);

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setPose(id, pose);
  }
  return success;
}

bool ViSlamBackend::setSpeedAndBias(StateId id, const SpeedAndBias &speedAndBias)
{
  bool success = realtimeGraph_.setSpeedAndBias(id, speedAndBias);

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setSpeedAndBias(id, speedAndBias);
  }
  return success;
}

bool ViSlamBackend::setExtrinsics(StateId id, uchar camIdx,
                                  const kinematics::TransformationCacheless & extrinsics)
{
  bool success = realtimeGraph_.setExtrinsics(id, camIdx, extrinsics);

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setExtrinsics(id, camIdx, extrinsics);
  }
  return success;
}

bool ViSlamBackend::isInImuWindow(StateId id) const
{
  return auxiliaryStates_.at(id).isImuFrame;
}

Time ViSlamBackend::timestamp(StateId id) const
{
  return realtimeGraph_.timestamp(id);
}

void ViSlamBackend::drawOverheadImage(cv::Mat &image, int idx) const
{
  const ViGraphEstimator& graph = idx==0 ? realtimeGraph_ : fullGraph_;
  static const double borderPixels = 50;

  Eigen::Vector3d min(-0.05,-0.05,-0.05);
  Eigen::Vector3d max(0.05,0.05,0.05);
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {
    const Eigen::Vector3d pos = iter->second.pose->estimate().r();
    for(int i=0; i<3; ++i) {
      if(pos[i] > max[i]) {
        max[i] = pos[i];
      }
      if(pos[i] < min[i]) {
        min[i] = pos[i];
      }
    }
  }
  const Eigen::Vector3d centre = 0.5*(max+min);
  const double scale = std::min((image.cols-2.0*borderPixels)/(max[0]-min[0]),
      (image.rows-2.0*borderPixels)/(max[1]-min[1]));

  for(auto iter = graph.landmarks_.begin(); iter != graph.landmarks_.end(); ++iter) {
    const Eigen::Vector4d hp = iter->second.hPoint->estimate();
    const Eigen::Vector3d pos = (hp.head<3>()/hp[3]-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    const double quality = iter->second.quality; 
    if(cvPos.x > 0.0 && cvPos.y > 0.0 && cvPos.x < (image.cols-1) && cvPos.y < (image.rows-1)) {
        cv::circle(image, cvPos, 1, cv::Scalar(0,std::min(255,20+int(quality/0.03*225.0)),0),
                                               cv::FILLED, cv::LINE_AA);
    }
  }

  for(auto iter = graph.states_.begin(); iter!= graph.states_.end(); ++iter){
    for(auto mIter = iter->second.GpsFactors.begin(); mIter != iter->second.GpsFactors.end(); ++mIter) {
      Eigen::Vector3d gps_point_in_W = (graph.T_GW().inverse().T() * mIter->errorTerm->measurement().homogeneous()).head<3>();
      Eigen::Vector3d gps_point_in_img = (gps_point_in_W - centre) * scale;
      cv::Point2d cvPos(gps_point_in_img[0]+image.cols*0.5, -gps_point_in_img[1]+image.rows*0.5);
      cv::drawMarker(image, cvPos, cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 12, 3);
    }
  }

  std::map<uint64_t, std::map<uint64_t, int>> coObservationCounts;
  for(auto iter=graph.landmarks_.begin(); iter!=graph.landmarks_.end(); ++iter) {
    auto obs = iter->second.observations;
    std::set<uint64> covisibilities;
    for(auto obsiter=obs.begin(); obsiter!=obs.end(); ++obsiter) {
      covisibilities.insert(obsiter->first.frameId);
    }
    for(auto i0=covisibilities.begin(); i0!=covisibilities.end(); ++i0) {
      for(auto i1=covisibilities.begin(); i1!=covisibilities.end(); ++i1) {
        if(*i1>=*i0) {
          continue;
        }
        if(coObservationCounts.find(*i0)==coObservationCounts.end()) {
          coObservationCounts[*i0][*i1] = 1;
        } else {
          if (coObservationCounts.at(*i0).find(*i1)==coObservationCounts.at(*i0).end()) {
            coObservationCounts.at(*i0)[*i1] = 1;
          } else {
            coObservationCounts.at(*i0).at(*i1)++;
          }
        }
      }
    }
  }

  for(auto i0=coObservationCounts.begin(); i0!=coObservationCounts.end(); ++i0) {
    for(auto i1=coObservationCounts.at(i0->first).begin();
        i1!=coObservationCounts.at(i0->first).end(); ++i1) {
      if(!graph.states_.count(StateId(i0->first))) continue;
      if(!graph.states_.count(StateId(i1->first))) continue;
      const Eigen::Vector3d p0 = graph.states_.at(StateId(i0->first)).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(StateId(i1->first)).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 50.0 + std::min(205, i1->second*2);
      cv::line(image, cvPos0, cvPos1, cv::Scalar(0,brightness, brightness), 1, cv::LINE_AA);
    }
  }

  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);

    for(auto piter=iter->second.twoPoseLinks.begin();
        piter != iter->second.twoPoseLinks.end(); piter++) {
      auto refId = piter->second.errorTerm->referencePoseId();
      auto otherId = piter->second.errorTerm->otherPoseId();
      if(refId<otherId) {

        const Eigen::Vector3d p0 = graph.states_.at(refId).pose->estimate().r();
        const Eigen::Vector3d p1 = graph.states_.at(otherId).pose->estimate().r();
        const Eigen::Vector3d pos0 = (p0-centre)*scale;
        const Eigen::Vector3d pos1 = (p1-centre)*scale;
        cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
        cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
        double brightness = 50.0 + std::min(205.0, piter->second.errorTerm->strength());
        cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2,brightness, brightness), 3,
                 cv::LINE_AA);
      }
    }

    for(auto piter=iter->second.twoPoseConstLinks.begin(); piter !=
        iter->second.twoPoseConstLinks.end(); piter++) {
      const Eigen::Vector3d p0 = graph.states_.at(iter->first).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(piter->first).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 200;
      cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2.0,brightness, brightness/2.0), 3,
               cv::LINE_AA);
    }
  }

  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,0,0);
    auto colour2 = cv::Scalar(255,255,255);
    if(!iter->second.isKeyframe) {
      colour1 = cv::Scalar(127,127,127);
    }
    if(idx<=2) {
        if(keyFrames_.count(iter->first)) {
          colour1 = cv::Scalar(255,255,0);
        }
        const bool isPoseGraphFrame = auxiliaryStates_.at(iter->first).isPoseGraphFrame;
        if(isPoseGraphFrame) {
          colour2 = cv::Scalar(127,127,127);
          if(iter->second.pose->fixed()) {
            colour2 = cv::Scalar(25,25,200);
          }
        }
    }

    if(iter->second.nextImuLink.errorTerm) {
      auto iterNext = iter;
      iterNext++;
      const Eigen::Vector3d nextPos = (iterNext->second.pose->estimate().r()-centre)*scale;
      cv::Point2d nextCvPos(nextPos[0]+image.cols*0.5, -nextPos[1]+image.rows*0.5);
      cv::line(image, cvPos, nextCvPos, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour2, 1, cv::LINE_AA);
    std::stringstream stream;
    stream << iter->first.value();
    cv::putText(image, stream.str(), cvPos+cv::Point2d(6,3),
                cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  }

  if(idx >2) {

      auto T_WS = graph.states_.rbegin()->second.pose->estimate();
      const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
      std::stringstream postext;
      postext << "position = ["
              << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
      cv::putText(image, postext.str(), cv::Point(15,15),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream veltext;
      veltext << "velocity = ["
              << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
      cv::putText(image, veltext.str(), cv::Point(15,35),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream gyroBiasText;
      gyroBiasText << "gyro bias = ["
                   << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
      cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream accBiasText;
      accBiasText << "acc bias = ["
                  << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
      cv::putText(image, accBiasText.str(), cv::Point(15,75),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      return;
  }

  std::map<StateId, size_t> counts;
  for(auto iter = loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    counts[*iter] = 0;
  }
  const StateId cId = currentStateId();
  for (auto pit = graph.landmarks_.begin(); pit != graph.landmarks_.end(); ++pit) {
    const auto& residuals = pit->second.observations;
    bool isObservedInNewFrame = false;
    std::set<uint64_t> loopClosureFramesObserved;
    for (const auto& r : residuals) {
      uint64_t poseId = r.first.frameId;
      if(cId.value() == poseId) {
        isObservedInNewFrame = true;
      }
      if(counts.find(StateId(poseId)) != counts.end()) {
        loopClosureFramesObserved.insert(poseId);
      }
    }

    if(isObservedInNewFrame) {
      for(auto iter = loopClosureFramesObserved.begin(); iter!=loopClosureFramesObserved.end();
          ++iter) {
        counts.at(StateId(*iter))++;
      }
    }
  }

  for(auto iter=loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    const Eigen::Vector3d pos = (graph.states_.at(*iter).pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,255);
    auto colour2 = cv::Scalar(255,0,0);

    cv::circle(image, cvPos, 5, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, int(6.0f+float(counts.at(*iter))/10.0f), colour2,
               int(1+counts.at(*iter))/5, cv::LINE_AA);
  }

  auto T_WS = graph.states_.rbegin()->second.pose->estimate();
  const Eigen::Vector3d pos = (T_WS.r()-centre)*scale;
  cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
  auto colour1 = cv::Scalar(127,127,127);
  auto colour2 = cv::Scalar(255,255,255);
  if(graph.states_.rbegin()->second.isKeyframe) {
    colour1 = cv::Scalar(255,255,0);
  }
  cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
  cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);

  StateId kfId = currentKeyframeStateId();
  if(kfId.isInitialised()) {
    kinematics::Transformation T_WS_kf = graph.states_.at(kfId).pose->estimate();
    const Eigen::Vector3d pos = (T_WS_kf.r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,127);
    auto colour2 = cv::Scalar(255,255,255);

    cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
  }

  const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
  std::stringstream postext;
  postext << "position = [" << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
  cv::putText(image, postext.str(), cv::Point(15,15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream veltext;
  veltext << "velocity = ["
          << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
  cv::putText(image, veltext.str(), cv::Point(15,35),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream gyroBiasText;
  gyroBiasText << "gyro bias = ["
               << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
  cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream accBiasText;
  accBiasText << "acc bias = ["
              << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
  cv::putText(image, accBiasText.str(), cv::Point(15,75),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}

bool ViSlamBackend::getObservedIds(StateId id, std::set<StateId> &observedIds) const
{
  auto iter = realtimeGraph_.states_.find(id);
  if (iter == realtimeGraph_.states_.end()) {
    return false;
  }

  const auto &state = iter->second;
  for (const auto &link : state.twoPoseLinks) {
    observedIds.insert(link.second.state0);
    observedIds.insert(link.second.state1);
  }

  if (keyFrames_.count(id) || imuFrames_.count(id) || loopClosureFrames_.count(id)) {
    std::set<LandmarkId> landmarks;
    for (const auto &obs : state.observations) {
      landmarks.insert(obs.second.landmarkId);
    }
    for (const auto &lmId : landmarks) {
      for (const auto &obs : realtimeGraph_.landmarks_.at(lmId).observations) {
        observedIds.insert(StateId(obs.first.frameId));
      }
    }
  }

  auto next = iter;
  next++;
  if (next != realtimeGraph_.states_.end()) {
    observedIds.insert(next->first);
  }
  if (iter != realtimeGraph_.states_.begin()) {
    auto previous = iter;
    previous--;
    observedIds.insert(previous->first);
  }

  return true;
}

bool ViSlamBackend::isLoopClosureFrame(StateId frameId) const {
  return loopClosureFrames_.count(frameId) > 0;
}

bool ViSlamBackend::isRecentLoopClosureFrame(StateId frameId) const {
  for(auto id : keyFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  for(auto id : loopClosureFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  return false;
}

void ViSlamBackend::addLoopClosureFrame(StateId loopClosureFrameId,
                                        std::set<LandmarkId> & loopClosureLandmarks,
                                        bool skipFullGraphOptimisation)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_,
                    "trying to add loop closure while loop closing")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "trying to add loop closure while loop closing")

  if(loopClosureFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add previously added loop-closure frame";
    return;
  }
  if(keyFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add current keyframe as loop-closure frame" << std::endl;
    return;
  }

  loopClosureFrames_.insert(loopClosureFrameId); 
  currentLoopClosureFrames_.insert(loopClosureFrameId);

  std::set<StateId> connectedStates;
  OKVIS_ASSERT_TRUE_DBG(Exception, auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame,
                    "must be pose graph frame")
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(
        loopClosureFrameId, &loopClosureLandmarks, &connectedStates, &allObservations);
  for(auto lm : loopClosureLandmarks) {
    const auto & landmark = realtimeGraph_.landmarks_.at(lm);
    if(!fullGraph_.landmarkExists(lm)) {
      fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
      fullGraph_.setLandmarkQuality(lm, landmark.quality);
    }
  }
  fullGraph_.removeTwoPoseConstLinks(loopClosureFrameId);
  for(const auto& obs : allObservations) {
    const bool useCauchy = obs.lossFunction != nullptr;
    LandmarkId landmarkId(obs.hPoint->id());
    fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                           obs.keypointIdentifier, useCauchy);
  }

  for(auto connectedState : connectedStates) {
    if(auxiliaryStates_.at(connectedState).isPoseGraphFrame
       && keyFrames_.count(connectedState)==0) {
      loopClosureFrames_.insert(connectedState);
      currentLoopClosureFrames_.insert(connectedState);
      auxiliaryStates_.at(connectedState).isPoseGraphFrame = false;
    } 
  }
  auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame = false;

  if(!loopClosureFrames_.empty()) {
    StateId oldestId = loopClosureFrameId;
    if(fullGraph_.states_.count(oldestId)) {
      StateId oldestIdToSetVariable = auxiliaryStates_.at(oldestId).loopId;

      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable);
          iter != auxiliaryStates_.end(); ++iter) {
        iter->second.loopId = oldestIdToSetVariable;
      }

      Time oldestT = fullGraph_.states_.at(oldestId).timestamp;
      int ctr = 0;
      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable); ; --iter) {
        if(ctr == numPoseGraphFrames || iter==auxiliaryStates_.begin()) {
          while((oldestT - fullGraph_.timestamp(iter->first)).toSec()<minDeltaT) {
            if(iter==auxiliaryStates_.begin()) {
              break;
            }
            --iter;
          }

          if (lastFreeze_.isInitialised()) {

            if (iter->first > lastFreeze_) {
              while (iter->first > lastFreeze_) {
                if (iter == auxiliaryStates_.begin()) {
                  break;
                }
                --iter;
              }
            }
          }

          fullGraph_.unfreezePosesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezePosesUntil(iter->first);
          }
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezeSpeedAndBiasesUntil(iter->first);
          }
          break;
        }
        if(iter==auxiliaryStates_.begin()) {
          fullGraph_.unfreezePosesFrom(iter->first);
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          break;
        }
        ctr++;
      }
    }
  }

  for(auto id : keyFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }
  for(auto id : loopClosureFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }

  if(!skipFullGraphOptimisation) {
    needsFullGraphOptimisation_ = true;
  }
}

void ViSlamBackend::addGpsAlignmentFrame(StateId gpsLossFrameId) {

  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "trying to add gps loop closure while loop closing")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "trying to add gps loop closure while loop closing")
  LOG(INFO) << "[GPS] Kicking off full graph optimisation due to gps loop closure, starting at state id = " << gpsLossFrameId.value()
            << " up to current state id  = " << fullGraph_.states_.rbegin()->first.value();

  StateId oldestIdToSetVariable = gpsLossFrameId;

  Time oldestT = fullGraph_.states_.at(oldestIdToSetVariable).timestamp;
  int ctr = 0;

  for(auto iter = fullGraph_.states_.find(oldestIdToSetVariable) ; ; --iter){
      if(ctr == numPoseGraphFrames || iter == fullGraph_.states_.begin()){
          while((oldestT - fullGraph_.timestamp(iter->first)).toSec()<minDeltaT) {
            if(iter==fullGraph_.states_.begin()) {
              break;
            }
            --iter;
          }
          fullGraph_.unfreezePosesFrom(iter->first);
          if(iter!=fullGraph_.states_.begin()) {
            fullGraph_.freezePosesUntil(iter->first);
          }
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          if(iter!=fullGraph_.states_.begin()) {
            fullGraph_.freezeSpeedAndBiasesUntil(iter->first);
          }
          break;
      }
      if(iter==fullGraph_.states_.begin()) {
        fullGraph_.unfreezePosesFrom(iter->first);
        fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
        break;
      }
      ctr++; 
  }

  needsFullGraphOptimisation_ = true;
}

bool ViSlamBackend::synchroniseRealtimeAndFullGraph(std::vector<StateId> &updatedStates)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "cannot synchronise while loop-closing")

  keyFrames_.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  loopClosureFrames_.clear();

  auto landmarks = fullGraph_.landmarks_;
  for(auto iter = landmarks.begin(); iter != landmarks.end(); ++iter) {
    if(realtimeGraph_.landmarks_.count(iter->first) == 0) {
      fullGraph_.removeLandmark(iter->first);
    }
  }

  StateId oldFrameId;
  for(auto iter = fullGraph_.states_.crbegin(); iter != fullGraph_.states_.crend(); ++iter) {
    if(realtimeGraph_.states_.count(iter->first)) {
      oldFrameId = iter->first;
      break;
    }
  }

  const kinematics::Transformation T_WS_old = realtimeGraph_.pose(oldFrameId);
  const kinematics::Transformation T_WS_new = fullGraph_.pose(oldFrameId);
  const kinematics::Transformation T_Wnew_Wold = T_WS_new * T_WS_old.inverse();

  for(const auto& addState : addStatesBacklog_) {
    bool asKeyframe = false;
    const bool existsInRealtimeGraph = realtimeGraph_.states_.count(addState.id) !=0;
    if(existsInRealtimeGraph) {
      if(realtimeGraph_.states_.at(addState.id).isKeyframe) {
        asKeyframe = true;
      }
    }
    fullGraph_.addStatesPropagate(addState.timestamp, addState.imuMeasurements, asKeyframe);
    kinematics::Transformation T_WS;
    SpeedAndBias speedAndBias;
    if(!existsInRealtimeGraph) {
      const auto& anystate = realtimeGraph_.anyState_.at(addState.id);
      const kinematics::Transformation T_Sk_S = anystate.T_Sk_S;
      const kinematics::Transformation T_WSk =
          T_Wnew_Wold*realtimeGraph_.states_.at(anystate.keyframeId).pose->estimate();
      T_WS = T_WSk*T_Sk_S;
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias.head<3>() = T_WSk.C() * anystate.v_Sk;
    } else {
      const kinematics::Transformation T_S0S1 =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse()*
          realtimeGraph_.states_.at(addState.id).pose->estimate();
      T_WS = fullGraph_.states_.at(oldFrameId).pose->estimate()*T_S0S1;
      const SpeedAndBias speedAndBias_old =
          realtimeGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      const Eigen::Vector3d v_S =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse().C()
          *speedAndBias_old.head<3>();
      speedAndBias.head<3>() = T_WS.C()*v_S;
    }
    fullGraph_.setPose(addState.id, T_WS);
    fullGraph_.setSpeedAndBias(addState.id, speedAndBias);
  }
  addStatesBacklog_.clear();

  for(const auto& eliminateState : eliminateStates_) {
    fullGraph_.removeAllObservations(eliminateState.first);

    fullGraph_.eliminateStateByImuMerge(eliminateState.first, eliminateState.second);

  }

  for(const auto& eliminateState : eliminateStates_) {
    const auto iter = fullGraph_.states_.find(eliminateState.second);
    if(iter != fullGraph_.states_.end()) {
      if(iter->second.previousImuLink.errorTerm) {
        if (fullGraph_.imuParametersVec_.at(0).use) {
          std::static_pointer_cast<ceres::ImuError>(iter->second.previousImuLink.errorTerm)
            ->syncFrom(*std::static_pointer_cast<ceres::ImuError>(
              realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        } else {
          std::static_pointer_cast<ceres::PseudoImuError>(iter->second.previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        }
      }
    }
  }
  eliminateStates_.clear();
  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {
      break;
    }
    auto & errorTerm = realtimeGraph_.states_.at(riter->first).previousImuLink.errorTerm;
    if(errorTerm) {
      if (fullGraph_.imuParametersVec_.at(0).use) {
        std::static_pointer_cast<ceres::ImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::PseudoImuError>(riter->second.previousImuLink.errorTerm));
      }
    }
  }

  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
      ++iter) {
    if(fullGraph_.landmarks_.count(iter->first) == 0) {
      realtimeGraph_.setLandmark(iter->first, T_Wnew_Wold * iter->second.hPoint->estimate());
      fullGraph_.addLandmark(
            iter->first, iter->second.hPoint->estimate(), iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  for(auto addGpsMeas : addGpsBacklog_){
      bool stillExistsInRealtimeGraph = fullGraph_.states_.count(addGpsMeas.id) !=0;
      if(stillExistsInRealtimeGraph) {
          if(addGpsMeas.reInitFlag){
            fullGraph_.reInitGpsExtrinsics();
            }
          fullGraph_.addGpsMeasurement(addGpsMeas.id, addGpsMeas.gpsMeasurement, addGpsMeas.imuMeasurements);
      }
  }
  addGpsBacklog_.clear();

  if(!fullGraph_.isGpsFixed()){
      const kinematics::Transformation T_GW_new = fullGraph_.T_GW(); 

      realtimeGraph_.setGpsExtrinsics(T_GW_new);
      if(gpsObservability_){ 
          LOG(INFO) << "[GPS] Fixing T_GW: \n" << fullGraph_.T_GW().T3x4();
          realtimeGraph_.setGpsExtrinsics(T_GW_new);
          fullGraph_.freezeGpsExtrinsics();
          realtimeGraph_.freezeGpsExtrinsics();
      }
  }

  for(auto alignmentTerm : addSubmapAlignmentBacklog_){
    bool stillExistsInRealtimeGraph = fullGraph_.states_.count(StateId(alignmentTerm.frame_B_id)) !=0;
    if(stillExistsInRealtimeGraph) {
      DLOG(INFO) << "reversely adding measurements to fullGraph at state " << alignmentTerm.frame_B_id
                << " and submap at " << alignmentTerm.frame_A_id << std::endl;
      fullGraph_.addSubmapAlignmentConstraints(alignmentTerm.submap_ptr, alignmentTerm.frame_A_id,
                                               alignmentTerm.frame_B_id, alignmentTerm.pointCloud,
                                               alignmentTerm.sensorError, alignmentTerm.isLidar,
                                               alignmentTerm.robustFunction);
    }
  }
  addSubmapAlignmentBacklog_.clear();

  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {

      OKVIS_ASSERT_TRUE(Exception, riter->second.pose->estimate().T()
                        ==realtimeGraph_.pose(riter->first).T(), "O-O")
      OKVIS_ASSERT_TRUE(Exception, riter->second.speedAndBias->estimate()
                        ==realtimeGraph_.speedAndBias(riter->first), "O-O")
      break;
    } else {
      updatedStates.push_back(riter->first);
    }
    if(realtimeGraph_.states_.count(riter->first) == 0) {
      OKVIS_THROW(Exception, "impossible: state not present")

      continue;
    }
    realtimeGraph_.setPose(riter->first, riter->second.pose->estimate());
    realtimeGraph_.setSpeedAndBias(riter->first, riter->second.speedAndBias->estimate());
    for(size_t i = 0; i<riter->second.extrinsics.size(); ++i) {
      if(!riter->second.extrinsics.at(i)->fixed()) {
        realtimeGraph_.setExtrinsics(riter->first, uchar(i),
                                     riter->second.extrinsics.at(i)->estimate());
      }
    }
  }

  for(auto iter = fullGraph_.landmarks_.begin(); iter != fullGraph_.landmarks_.end(); ++iter) {
    OKVIS_ASSERT_TRUE_DBG(Exception, realtimeGraph_.landmarkExists(iter->first), "not allowed")
    realtimeGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                               iter->second.hPoint->initialized());
    realtimeGraph_.setLandmarkQuality(iter->first, iter->second.quality);
  }

  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmark full vs realtime graph")
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }

    auto observations = fullGraph_.landmarks_.at(lm).observations;
    for(const auto & obs : observations) {
      fullGraph_.removeObservation(obs.first);
    }
  }
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }

    for(const auto & obs : realtimeGraph_.landmarks_.at(lm).observations) {
      fullGraph_.addExternalObservation(obs.second.errorTerm, lm, obs.first, true);
    }
  }

  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; 
    }
    auto relativePoselinks = fullGraph_.states_.at(stateId).relativePoseLinks;
    for(const auto& link : relativePoselinks) {
      fullGraph_.removeRelativePoseConstraint(link.second.state0, link.second.state1);
    }
    auto twoPoselinks = fullGraph_.states_.at(stateId).twoPoseConstLinks;
    for(const auto& link : twoPoselinks) {
      fullGraph_.removeTwoPoseConstLink(link.second.state0, link.second.state1);
    }
  }

  std::set<::ceres::ResidualBlockId> addedRelativePoseLinks;
  std::set<::ceres::ResidualBlockId> addedTwoPoseLinks;
  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; 
    }
    for(const auto& relPoseLink : realtimeGraph_.states_.at(stateId).relativePoseLinks) {
      if(addedRelativePoseLinks.count(relPoseLink.second.residualBlockId)==0) {
        fullGraph_.addRelativePoseConstraint(
              relPoseLink.second.state0, relPoseLink.second.state1,
              relPoseLink.second.errorTerm->T_AB(),
              relPoseLink.second.errorTerm->information());
        addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
      }
    }
    for(const auto& twoPoseLink : realtimeGraph_.states_.at(stateId).twoPoseLinks) {
      if(addedTwoPoseLinks.count(twoPoseLink.second.residualBlockId)==0) {
        fullGraph_.addExternalTwoPoseLink(
              twoPoseLink.second.errorTerm->cloneTwoPoseGraphErrorConst(),
              twoPoseLink.second.state0, twoPoseLink.second.state1);
        addedTwoPoseLinks.insert(twoPoseLink.second.residualBlockId);
      }
    }
  }
  touchedStates_.clear();
  touchedLandmarks_.clear();

  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmarks full vs realtime graph")
  OKVIS_ASSERT_TRUE(Exception, fullGraph_.states_.size() == realtimeGraph_.states_.size(),
                    "inconsistent states full vs realtime graph")

  currentLoopClosureFrames_.clear();
  isLoopClosureAvailable_ = false;

  return true;
}

int ViSlamBackend::cleanUnobservedLandmarks() {

  std::map<LandmarkId, std::set<KeypointIdentifier>> removed;
  int removed1 = realtimeGraph_.cleanUnobservedLandmarks(&removed);
  for(const auto & rem : removed) {
    for(const auto & obs : rem.second) {

      multiFrame(StateId(obs.frameId))->setLandmarkId(obs.cameraIndex, obs.keypointIndex, 0);
    }
  }
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & rem : removed) {
      touchedLandmarks_.insert(rem.first);
      for(auto obs : rem.second) {

        touchedStates_.insert(StateId(obs.frameId));
      }
    }
    return removed1; 
  } else {
    int removed0 = fullGraph_.cleanUnobservedLandmarks();
    OKVIS_ASSERT_TRUE(Exception, removed0 == removed1, "landmarks cleaned inconsistent!")
    return removed1;
  }

}

bool ViSlamBackend::mergeLandmark(const LandmarkId &fromId, const LandmarkId &intoId)
{
  bool success = (realtimeGraph_.mergeLandmark(fromId, intoId, multiFrames_));

  auto observations = realtimeGraph_.landmarks_.at(intoId).observations;
  for (const auto &observation : observations) {
    multiFrames_.at(StateId(observation.first.frameId))
      ->setLandmarkId(observation.first.cameraIndex,
                      observation.first.keypointIndex,
                      intoId.value());
  }

  if (isLoopClosing_ || isLoopClosureAvailable_) {
    for (const auto &obs : observations) {
      touchedStates_.insert(StateId(obs.first.frameId));
    }
    touchedLandmarks_.insert(fromId);
    touchedLandmarks_.insert(intoId);
  } else {

    success &= fullGraph_.mergeLandmark(fromId, intoId, multiFrames_);
  }

  return success;
}

int ViSlamBackend::mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, fromIds.size() == intoIds.size(), "vectors must be same lengths")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "loop closure not finished, cannot merge landmarks")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  std::map<LandmarkId, LandmarkId> changes; 
  int ctr = 0;
  for(size_t i = 0; i < fromIds.size(); ++i) {

    while(changes.count(fromIds.at(i))) {
      fromIds.at(i) = changes.at(fromIds.at(i));
    }

    while(changes.count(intoIds.at(i))) {
      intoIds.at(i) = changes.at(intoIds.at(i));
    }

    if(fromIds.at(i) == intoIds.at(i)) {
      continue; 
    }

    if(realtimeGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_)) {
      ctr++;
    }
    fullGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_);
    changes[fromIds.at(i)] = intoIds.at(i);

    auto observations = realtimeGraph_.landmarks_.at(intoIds.at(i)).observations;
    for(const auto & observation : observations) {
      multiFrames_.at(StateId(observation.first.frameId))->setLandmarkId(
            observation.first.cameraIndex, observation.first.keypointIndex, intoIds.at(i).value());
    }
  }

  return ctr;
}

void ViSlamBackend::optimiseFullGraph(int numIter, ::ceres::Solver::Summary &summary,
                                      int numThreads, bool verbose,
                                      const FinalBaProgressCallback& progressCallback,
                                      const int progressStage)
{

  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "can't optimise posegraph; already loop-closing")
  DLOG(INFO) << "Starting fullGraph_ optimisation.";
  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = true;
  if(!fullGraphRelativePoseConstraints_.empty()) {
    kinematics::Transformation T_SS_measured;
    StateId pose_i, pose_j;
    for(auto & item : fullGraphRelativePoseConstraints_) {
      T_SS_measured = item.T_Si_Sj;
      pose_i = item.pose_i;
      pose_j = item.pose_j;
      fullGraph_.addRelativePoseConstraint(item.pose_i, item.pose_j, item.T_Si_Sj,
                                           100*item.information);
    }
    fullGraph_.options().function_tolerance = 0.001;
    fullGraph_.optimise(numIter/3, numThreads, verbose);
    for(auto & item : fullGraphRelativePoseConstraints_) {
      fullGraph_.removeRelativePoseConstraint(item.pose_i, item.pose_j);
    }
    fullGraphRelativePoseConstraints_.clear();
  }
  fullGraph_.options().function_tolerance = 1e-6;

  std::unique_ptr<FinalBaIterationCallback> progress;
  if (progressCallback && progressStage > 0 && numIter > 0) {
    progress = std::make_unique<FinalBaIterationCallback>(
        progressCallback, progressStage, numIter);
    progressCallback(progressStage, 0, numIter);
  }
  ScopedSolverCallback callback_guard(fullGraph_.options(), progress.get());
  fullGraph_.optimise(numIter, numThreads, verbose);
  summary = fullGraph_.summary();
  fullGraph_.updateLandmarks(); 

  isLoopClosureAvailable_ = true;
  isLoopClosing_ = false;
}

void ViSlamBackend::doFinalBa(
    int numIter, ::ceres::Solver::Summary &summary,
    std::set<StateId> & updatedStatesBa, double extrinsicsPositionUncertainty,
    double extrinsicsOrientationUncertainty, int numThreads, bool verbose,
    const FinalBaProgressCallback& progressCallback)
{

  int i=0;
  for(const auto & state : fullGraph_.states_) {
    if(state.second.twoPoseConstLinks.size()>0) {
      if(keyFrames_.count(state.first) == 0 && loopClosureFrames_.count(state.first) == 0) {
        keyFrames_.insert(state.first);
      }
      std::cout << "\rConstructing VI-BA problem... "
                << std::round(10000.0*double(i)/double(fullGraph_.states_.size()))/100.0 << "%";
      expandKeyframe(state.first);
      i++;
    }
  }
  fullGraph_.cleanUnobservedLandmarks();
  std::cout << "\rConstructing VI-BA problem... " << "100.00%" << std::endl;

  fullGraph_.unfreezePosesFrom(StateId(1));
  fullGraph_.unfreezeSpeedAndBiasesFrom(StateId(1));

  if(!fullGraph_.gpsParametersVec_.empty()){
    fullGraph_.unfreezeGpsExtrinsics();
  }

  ceres::ImuError::redoPropagationAlways = true;

  std::cout << "Running optimisation (verbose="
            << (verbose?"true)":"false)...") << std::endl;
  optimiseFullGraph(
      numIter, summary, numThreads, verbose, progressCallback, 1);

  fullGraph_.removeSpeedAndBiasPrior(StateId(1));

  if(extrinsicsPositionUncertainty > 0.0 && extrinsicsOrientationUncertainty > 0.0) {
    std::cout << "Running optimisation again without priors & fixation (verbose="
              << (verbose?"true)":"false)...") << std::endl;
    fullGraph_.setExtrinsicsVariable();
    fullGraph_.softConstrainExtrinsics(
          extrinsicsPositionUncertainty, extrinsicsOrientationUncertainty);
  } else {
    std::cout << "Running optimisation again without priors (verbose="
              << (verbose?"true)":"false)...") << std::endl;
  }

  optimiseFullGraph(
      numIter, summary, numThreads, verbose, progressCallback, 2);

  cv::Mat img=cv::Mat::zeros(2000,2000,CV_8UC3);
  drawOverheadImage(img, 2);
  cv::imwrite("fullBa.png", img);

  const auto & state = fullGraph_.states_.rbegin()->second;
  for(size_t i = 0; i < state.extrinsics.size(); ++i) {
    std::cout << "extrinsics T_SC" << i << ":" << std::endl;
    std::cout << std::setprecision(15) << state.extrinsics.at(i)->estimate().T() << std::endl;
  }
  Eigen::Matrix<double,6,1> biases = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    biases += s.second.speedAndBias->estimate().tail<6>();
  }
  biases = (1.0/double(fullGraph_.states_.size()))*biases;
  std::cout << "biases: " << std::setprecision(15) << biases.transpose() << std::endl;
  Eigen::Matrix<double,6,1> biasesStd = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    Eigen::Matrix<double,6,1> diff = (s.second.speedAndBias->estimate().tail<6>()-biases);
    biasesStd += diff.cwiseAbs2();
  }
  std::cout << "biases stdev: " << biasesStd.cwiseSqrt().transpose() << std::endl;

  std::vector<Eigen::Matrix<int, 12, 1>, Eigen::aligned_allocator<Eigen::Matrix<int, 12, 1>>> errorHistograms;
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    Eigen::Matrix<int, 12, 1> errHist;
    errHist.setZero();
    errorHistograms.push_back(errHist);
  }

  std::vector<cv::Mat> images;
  std::vector<cv::Mat> outliers;
  std::vector<int> inlierCtrs;
  std::vector<int> outlierCtrs;
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    inlierCtrs.push_back(0);
    outlierCtrs.push_back(0);
    images.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
    outliers.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
  }

  for(const auto & obs : fullGraph_.observations_) {
    auto frame = multiFrames_.at(StateId(obs.first.frameId));
    cv::KeyPoint kpt;
    frame->getCvKeypoint(obs.first.cameraIndex, obs.first.keypointIndex, kpt);
    double* params[3];
    params[0] = fullGraph_.states_.at(StateId(obs.first.frameId)).pose->parameters();
    params[1] = fullGraph_.landmarks_.at(obs.second.landmarkId).hPoint->parameters();
    params[2] = fullGraph_.states_.at(StateId(obs.first.frameId)).extrinsics.at(
          obs.first.cameraIndex)->parameters();
    Eigen::Vector2d err;
    obs.second.errorTerm->Evaluate(params, err.data(), nullptr);
    double badness = sqrt(err.transpose()*err)/3.0; 
    if(badness>1.0) {
      cv::circle(outliers.at(obs.first.cameraIndex), kpt.pt, 1, cv::Scalar(255,0,0),
                 cv::FILLED, cv::LINE_AA);
      outlierCtrs.at(obs.first.cameraIndex)++;
    } else {
      cv::Scalar colour(0,std::max(0.0,255.0*(1.0-badness)),std::min(255.0,255.0*badness));
      cv::circle(images.at(obs.first.cameraIndex), kpt.pt, 1, colour, cv::FILLED, cv::LINE_AA);
      inlierCtrs.at(obs.first.cameraIndex)++;
    }

    double errNorm = err.norm();
    int idx = static_cast<int> (std::floor(errNorm)) + 6;
    if(idx > 11)
      idx = 11;
    else if(idx < 0)
      idx = 0;
    errorHistograms.at(obs.first.cameraIndex)[idx] += 1;

  }

  std::vector<StateId> updatedStates;
  synchroniseRealtimeAndFullGraph(updatedStates);
  for(const auto & state : fullGraph_.states_) {
    updatedStatesBa.insert(state.first);
  }

  cleanUnobservedLandmarks();
}

bool ViSlamBackend::saveMap(std::string path)
{

  std::string g2oPath = path.substr(0, path.size() - 4) + ".g2o";
  Component component(fullGraph_.imuParametersVec_[0],
                      multiFrames_.at(StateId(1))->cameraSystem(),
                      fullGraph_,
                      multiFrames_);
  component.save(g2oPath);

  std::ofstream file(path);
  if(!file.good()) {
    return false;
  }

  std::ios init(nullptr);
  init.copyfmt(file);

  file << "landmarks:" << std::endl;
  for(const auto & landmark : fullGraph_.landmarks_) {
    if(landmark.second.quality > 0.001) {

      file << landmark.first.value() << ",";

      Eigen::Vector4d hposition = landmark.second.hPoint->estimate();
      Eigen::Vector3d position = hposition.head<3>()/hposition[3];
      file << position[0] << "," << position[1] << "," << position[2];
      file << std::endl;
    }
  }

  fullGraph_.computeCovisibilities();
  for(const auto & state : fullGraph_.states_) {
    file << "frame: " << state.first.value() << ", covisibilities: ";
    std::vector<std::pair<StateId, int>> covisibilities;
    for(const auto & state2 : fullGraph_.states_) {
      int c = fullGraph_.covisibilities(state.first, state2.first);
      if(c > 0) {
        covisibilities.push_back(std::make_pair(state2.first, c));
      }
    }
    std::sort(covisibilities.begin(), covisibilities.end(), 
         [](const std::pair<StateId, int> & left, const std::pair<StateId, int> & right){
           return left.second > right.second;});
    for(const auto & covisibility : covisibilities) {
      file << covisibility.first.value() << " ";
    }
    file << std::endl;
    for(const auto & obs : state.second.observations) {
      const auto & landmark = fullGraph_.landmarks_.at(obs.second.landmarkId);
      if(landmark.quality > 0.001) {
        file << obs.first.keypointIndex << "," << obs.second.landmarkId.value() << ","; 
        Eigen::Vector4d hposition = landmark.hPoint->estimate();
        Eigen::Vector3d position = hposition.head<3>()/hposition[3];
        file << position[0] << "," << position[1] << "," << position[2] << ","; 

        const unsigned char* descriptor =
            multiFrames_.at(state.first)->keypointDescriptor(
              obs.first.cameraIndex, obs.first.keypointIndex);
        for(size_t i=0; i<48; ++i) {
          file << std::setfill('0') << std::setw(2) << std::hex << uint32_t(descriptor[i]);
        }
        file.copyfmt(init); 
        file << std::endl;
      }
    }
  }

  return true;
}

bool ViSlamBackend::writeFinalCsvTrajectory(const std::string &csvFileName, bool rpg) const {
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success = csvFile.good();
  if (!success) {
    return false;
  }

  if (rpg) {
    csvFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
  } else {
    csvFile << "timestamp" << ", " << "p_WS_W_x" << ", " << "p_WS_W_y" << ", "
            << "p_WS_W_z" << ", " << "q_WS_x" << ", " << "q_WS_y" << ", "
            << "q_WS_z" << ", " << "q_WS_w" << ", " << "v_WS_W_x" << ", "
            << "v_WS_W_y" << ", " << "v_WS_W_z" << ", " << "b_g_x" << ", "
            << "b_g_y" << ", " << "b_g_z" << ", " << "b_a_x" << ", " << "b_a_y"
            << ", " << "b_a_z" << ", " << "NrGps" << ", " << "SID" << ", " << "gpsMode" << std::endl;
  }
  for (auto iter = realtimeGraph_.anyState_.begin(); iter != realtimeGraph_.anyState_.end(); ++iter) {
    Eigen::Vector3d p_WS_W;
    Eigen::Quaterniond q_WS;
    SpeedAndBias speedAndBiases;
    std::stringstream time;
    size_t nrGps;
    size_t gpsMode;
    StateId sid;

    if (iter->second.keyframeId.isInitialised()) {

      const ViGraph::State &keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      kinematics::Transformation T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      p_WS_W = T_WS.r();
      q_WS = T_WS.q();
      speedAndBiases.head<3>() = keyframeState.pose->estimate().C() * iter->second.v_Sk;
      speedAndBiases.tail<6>() = keyframeState.speedAndBias->estimate().tail<6>();
      time << iter->second.timestamp.sec << std::setw(9) << std::setfill('0')
            << iter->second.timestamp.nsec;
      nrGps = keyframeState.GpsFactors.size();
      gpsMode = keyframeState.gpsMode;
      sid = iter->first;
    } else {

      const ViGraph::State &state = realtimeGraph_.states_.at(iter->first);
      p_WS_W = state.pose->estimate().r();
      q_WS = state.pose->estimate().q();
      speedAndBiases = state.speedAndBias->estimate();
      time << state.timestamp.sec << std::setw(9) << std::setfill('0')
            << state.timestamp.nsec;
      nrGps = state.GpsFactors.size();
      gpsMode = state.gpsMode;
      sid = iter->first;
    }
    if (rpg) {
      csvFile << std::setprecision(19) << iter->second.timestamp.toSec() << " "
              << p_WS_W[0] << " " << p_WS_W[1] << " " << p_WS_W[2] << " "
              << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " " << q_WS.w() << std::endl;
    } else {
      csvFile << time.str() << ", " << std::scientific
              << std::setprecision(18) << p_WS_W[0] << ", " << p_WS_W[1] << ", "
              << p_WS_W[2] << ", " << q_WS.x() << ", " << q_WS.y() << ", "
              << q_WS.z() << ", " << q_WS.w() << ", " << speedAndBiases[0] << ", "
              << speedAndBiases[1] << ", " << speedAndBiases[2] << ", "
              << speedAndBiases[3] << ", " << speedAndBiases[4] << ", "
              << speedAndBiases[5] << ", " << speedAndBiases[6] << ", "
              << speedAndBiases[7] << ", " << speedAndBiases[8] << ", "
              << nrGps << ", " << sid.value() << ", " << gpsMode << ", "
              << iter->second.keyframeId.value() << std::endl;
    }
  }

  csvFile.close();
  return success;
}

bool ViSlamBackend::getFinalTrajectoryPositions(
    std::vector<FinalTrajectoryPosition>& positions) const
{
  positions.clear();
  positions.reserve(realtimeGraph_.anyState_.size());

  for (auto iter = realtimeGraph_.anyState_.begin(); iter != realtimeGraph_.anyState_.end();
       ++iter) {
    Eigen::Vector3d p_WS_W;
    Eigen::Quaterniond q_WS;

    if (iter->second.keyframeId.isInitialised()) {

      const ViGraph::State& keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      kinematics::Transformation T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      p_WS_W = T_WS.r();
      q_WS = T_WS.q();
    } else {
      const ViGraph::State& state = realtimeGraph_.states_.at(iter->first);
      p_WS_W = state.pose->estimate().r();
      q_WS = state.pose->estimate().q();
    }

    FinalTrajectoryPosition pose;
    pose.state_id = iter->first.value();
    pose.timestamp_ns = iter->second.timestamp.toNSec();
    pose.px = p_WS_W.x();
    pose.py = p_WS_W.y();
    pose.pz = p_WS_W.z();
    pose.qx = q_WS.x();
    pose.qy = q_WS.y();
    pose.qz = q_WS.z();
    pose.qw = q_WS.w();
    positions.push_back(pose);
  }

  return true;
}

bool ViSlamBackend::writeGlobalCsvTrajectory(const std::string &csvFileName) const
{
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success =  csvFile.good();
  if(!success) {
    return false;
  }

  kinematics::Transformation T_GW;

  csvFile << "timestamp" << ", " << "p_GA_G_x" << ", " << "p_GA_G_y" << ", "
               << "p_GA_G_z" << std::endl;
  for(auto iter=realtimeGraph_.anyState_.begin(); iter!=realtimeGraph_.anyState_.end(); ++iter) {
    Eigen::Vector3d p_GA_G;
    std::stringstream time;
    kinematics::Transformation T_WS;

    if(iter->second.keyframeId.isInitialised()) {

      const ViGraph::State& keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      time << iter->second.timestamp.sec << std::setw(9) << std::setfill('0')
           << iter->second.timestamp.nsec;
    } else {

      const ViGraph::State& state = realtimeGraph_.states_.at(iter->first);
      T_WS = state.pose->estimate();
      time << state.timestamp.sec << std::setw(9) << std::setfill('0')
           << state.timestamp.nsec;
    }

    T_GW = realtimeGraph_.T_GW();
    p_GA_G =  T_GW.C() * (T_WS.r() + T_WS.C()* realtimeGraph_.gpsParametersVec_.back().r_SA ) + T_GW.r();
    csvFile << time.str() << ", " << std::scientific
        << std::setprecision(18) << p_GA_G[0] << ", " << p_GA_G[1] << ", "
        << p_GA_G[2] <<  std::endl;
  }

  csvFile.close();
  return success;
}

bool ViSlamBackend::attemptLoopClosure(StateId pose_i, StateId pose_j,
                                       const kinematics::Transformation& T_Si_Sj,
                                       const Eigen::Matrix<double, 6, 6>& information,
                                       bool & skipFullGraphOptimisation,
                                       double driftPercentageHeuristic)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  if(realtimeGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(realtimeGraph_.states_.count(pose_j) == 0) {
    return false;
  }

  int numSteps = 0; 
  StateId lastLoopId = auxiliaryStates_.at(pose_i).loopId;
  for(auto iter = realtimeGraph_.states_.find(pose_i);
      iter != realtimeGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    if(lastLoopId!=loopId) {
      lastLoopId = loopId;
      numSteps++;
    }
  }
  StateId firstId = auxiliaryStates_.at(pose_i).loopId;
  lastLoopId = firstId;
  double distanceTravelled = 0.0;
  double distanceTravelled2 = 0.0; 
  kinematics::Transformation T_WS_i = realtimeGraph_.states_.at(pose_i).pose->estimate();
  kinematics::Transformation T_WS_last = realtimeGraph_.states_.at(pose_i).pose->estimate();
  auto iter = realtimeGraph_.states_.find(pose_i);
  std::vector<double> distances;
  ++iter;
  Eigen::Vector3d distanceTravelledVec(0.0,0.0,0.0);
  if(pose_i != lastLoopId) {
    distanceTravelledVec +=
        (realtimeGraph_.states_.at(lastLoopId).pose->estimate().r()-T_WS_i.r());
    distanceTravelled2 += distanceTravelledVec.norm();

  }
  for(; iter != realtimeGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    kinematics::Transformation T_WS_j = iter->second.pose->estimate();
    T_WS_last = T_WS_j;
    if(lastLoopId!=loopId) {
      const Eigen::Vector3d dsVec = (T_WS_j.r()-T_WS_i.r());
      const double ds = dsVec.norm();
      lastLoopId = loopId;
      distances.push_back(ds);
      distanceTravelledVec += dsVec;
      distanceTravelled += ds;
      distanceTravelled2 += ds;
      T_WS_i = T_WS_j;
    }
  }

  skipFullGraphOptimisation = false;
  {

    const kinematics::Transformation T_WSi = realtimeGraph_.pose(pose_i);
    const kinematics::Transformation T_WSj_old= realtimeGraph_.pose(pose_j);
    const kinematics::Transformation T_Si_Sj_old = T_WSi.inverse() * T_WSj_old;
    const kinematics::Transformation T_WSj_new = T_WSi * T_Si_Sj;
    kinematics::Transformation T_Wnew_Wold_final = T_WSj_new * T_WSj_old.inverse();

    Eigen::AngleAxisd aa(T_Wnew_Wold_final.q());
    aa.angle() = aa.angle() * (1.0 / double(numSteps));
    const kinematics::Transformation T_WW(Eigen::Vector3d(0.0,0.0,0.0), Eigen::Quaterniond(aa));

    kinematics::Transformation T_WS_prev = realtimeGraph_.pose(pose_i);
    kinematics::Transformation T_WS = realtimeGraph_.pose(pose_i);
    auto iter = realtimeGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    for( ; iter != realtimeGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {
        T_WS = T_WW * T_WS * T_SS;
        lastLoopId = loopId;
      } else {
        T_WS = T_WS * T_SS;
      }
    }

    const Eigen::Vector3d dr_W = T_WSj_new.r()-T_WS.r();

    const double relPositionError = dr_W.norm()/(distanceTravelled2);
    const double relOrientationError =
        T_WSj_new.q().angularDistance(T_WSj_old.q())/double(numSteps);
    const double relPositionErrorBudget = 
            driftPercentageHeuristic/100.0 + 
            0.02*distanceTravelledVec.norm()/distanceTravelled2 + 
            0.08/sqrt(numSteps); 
    const double relOrientationErrorBudget =
        0.0004 + 0.004/sqrt(numSteps); 
    if(relPositionError > relPositionErrorBudget
        || relOrientationError > relOrientationErrorBudget
        || numSteps < 1) {

      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (heuristic consistency).";
      LOG(INFO) << "Rel. pos. err. " << relPositionError << " vs budget "
          << relPositionErrorBudget << " m/m, rel. or. err. "
          << relOrientationError << " vs budget "
          << relOrientationErrorBudget << " rad/kf";
      LOG(INFO) << "dist. travelled " << distanceTravelled2 << " m, no. steps " << numSteps;

      return false;
    }

    Eigen::Matrix<double, 6, 6> P;
    okvis::PseudoInverse::symm(information, P);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(P.topLeftCorner<3, 3>());
    Eigen::Vector3d eigenvalues = saes.eigenvalues();
    double sigma = sqrt(eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
    if (sigma > 0.1 && 3.0 * sigma > relPositionErrorBudget * distanceTravelled2) {
      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (reloc. 3-sigma = " << 3.0*sigma << " m vs budget "
                   << relPositionErrorBudget*distanceTravelled2 << ") ";
      return false;
    }

    T_WS_prev = realtimeGraph_.pose(pose_i);
    T_WS = realtimeGraph_.pose(pose_i);
    iter = realtimeGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    int ctr = 0;
    double r = 0.0;
    for( ; iter != realtimeGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {

        r +=  distances.at(size_t(ctr)) / distanceTravelled;
        T_WS = T_WW *  T_WS * T_SS;
        lastLoopId = loopId;
        ++ctr;
      } else {
        T_WS = T_WS * T_SS;
      }
      const kinematics::Transformation T_WS_set(T_WS.r() + r*dr_W, T_WS.q());
      const kinematics::Transformation T_Wnew_Wold = T_WS_set * T_WSk_old.inverse();
      SpeedAndBias speedAndBias = iter->second.speedAndBias->estimate();
      const Eigen::Vector3d v_Wold = speedAndBias.head<3>();
      speedAndBias.head<3>() = T_Wnew_Wold.C() * v_Wold;
      realtimeGraph_.setPose(iter->first, T_WS_set);
      fullGraph_.setPose(iter->first, T_WS_set);
      realtimeGraph_.setSpeedAndBias(iter->first, speedAndBias);
      fullGraph_.setSpeedAndBias(iter->first, speedAndBias);
      updatedStatesLoopClosureAttempt_.insert(iter->first);
    }

    for(auto iter = realtimeGraph_.landmarks_.begin();
        iter != realtimeGraph_.landmarks_.end(); ++iter) {

      Eigen::Vector4d hPointNew = T_Wnew_Wold_final * iter->second.hPoint->estimate();
      realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
      fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    }
  }

  auxiliaryStates_.at(pose_j).closedLoop = true;

  fullGraphRelativePoseConstraints_.push_back(RelPoseInfo{T_Si_Sj, information, pose_i, pose_j});

  return true;
}

bool ViSlamBackend::attemptFullGpsAlignment(StateId pose_i , StateId pose_j, const okvis::kinematics::Transformation& T_GW_new){
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "loop closure not finished, cannot merge landmarks")

  if(realtimeGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(realtimeGraph_.states_.count(pose_j) == 0) {
    return false;
  }

  okvis::kinematics::Transformation T_GW_old = realtimeGraph_.T_GW(pose_i);
  LOG(INFO) << "[GPS Loop Closure] Between states "
            << pose_i.value() << " and " << pose_j.value() << " while most recent state is "
            << realtimeGraph_.states_.rbegin()->first.value() << "\n"
            << "T_GW_old:\n" << T_GW_old.T3x4() << "\n"
            << "T_GW_new:\n" << T_GW_new.T3x4();

  okvis::kinematics::Transformation T_WSj = realtimeGraph_.pose(pose_j);
  okvis::kinematics::Transformation T_WSi = realtimeGraph_.pose(pose_i);

  const kinematics::Transformation T_Wold_Wnew_final = T_GW_old.inverse() * T_GW_new;
  const kinematics::Transformation T_WSj_old = T_Wold_Wnew_final * T_WSj;

  int numVariableStates = 0; 
  for(auto iter = realtimeGraph_.states_.find(pose_i);
      iter != realtimeGraph_.states_.end(); ++iter) {
    numVariableStates++;
  }

  StateId firstId = auxiliaryStates_.at(pose_i).loopId;
  StateId lastId;
  StateId id;
  lastId = firstId;

  double distanceTravelled = 0.0;
  kinematics::Transformation T_WS_i = realtimeGraph_.states_.at(pose_i).pose->estimate();
  auto iter = realtimeGraph_.states_.find(pose_i);
  std::vector<double> distances;
  ++iter;
  for(; iter != realtimeGraph_.states_.end(); ++iter) {
    kinematics::Transformation T_WS_j = iter->second.pose->estimate();
    const double ds = (T_WS_j.r()-T_WS_i.r()).norm();
    distances.push_back(ds);
    distanceTravelled += ds;
    T_WS_i = T_WS_j;
  }

  Eigen::AngleAxisd aa(T_Wold_Wnew_final.q());
  aa.angle() = aa.angle() * (1.0 / double(numVariableStates));
  const kinematics::Transformation T_WW(Eigen::Vector3d(0.0,0.0,0.0), Eigen::Quaterniond(aa));
  kinematics::Transformation T_WS_prev = realtimeGraph_.pose(pose_i);
  kinematics::Transformation T_WS = realtimeGraph_.pose(pose_i);
  iter = realtimeGraph_.states_.find(pose_i);
  iter++;
  for( ; iter != realtimeGraph_.states_.end(); ++iter) {
    const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
    const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
    T_WS_prev = T_WSk_old;
    T_WS = T_WW *  T_WS * T_SS; 
  }

  const Eigen::Vector3d dr_W = T_WSj_old.r()-T_WS.r();

  T_WS_prev = realtimeGraph_.pose(pose_i);
  T_WS = realtimeGraph_.pose(pose_i);
  iter = realtimeGraph_.states_.find(pose_i);
  iter++;
  int ctr = 0;
  double r = 0.0;
  for( ; iter != realtimeGraph_.states_.end(); ++iter) {
    const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
    const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
    T_WS_prev = T_WSk_old;

    r +=  distances.at(ctr) / distanceTravelled;
    T_WS = T_WW * T_WS * T_SS; 
    ++ctr;
    const kinematics::Transformation T_WS_set(T_WS.r() + r*dr_W, T_WS.q());
    const kinematics::Transformation T_Wnew_Wold = T_WS_set * T_WSk_old.inverse(); 
    SpeedAndBias speedAndBias = iter->second.speedAndBias->estimate();
    speedAndBias.head<3>() = (T_Wnew_Wold.C() * speedAndBias.head<3>()).eval();
    realtimeGraph_.setPose(iter->first, T_WS_set);
    fullGraph_.setPose(iter->first, T_WS_set);
    realtimeGraph_.setSpeedAndBias(iter->first, speedAndBias);
    fullGraph_.setSpeedAndBias(iter->first, speedAndBias);
  }

  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end(); ++iter) {
    Eigen::Vector4d hPointNew = T_Wold_Wnew_final * iter->second.hPoint->estimate();
    realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
  }

  return true;
}

bool ViSlamBackend::attemptPosGpsAlignment(StateId pose_i , StateId pose_j, const Eigen::Vector3d& posAlignVec){

    OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
    OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "loop closure not finished, cannot merge landmarks")

    if(realtimeGraph_.states_.count(pose_i) == 0) {
      return false;
    }
    if(realtimeGraph_.states_.count(pose_j) == 0) {
      return false;
    }
    Eigen::Vector3d r_Wold_Wnew = posAlignVec;

    StateId oldestId=pose_i;

    fullGraph_.unfreezePosesFrom(pose_i);
    fullGraph_.freezePosesUntil(pose_i);

    fullGraph_.unfreezeSpeedAndBiasesFrom(pose_i);
    fullGraph_.freezeSpeedAndBiasesUntil(pose_i);

    int numVariableStates = 0; 
    for(auto iter = realtimeGraph_.states_.find(oldestId);
        iter != realtimeGraph_.states_.end(); ++iter) {
      numVariableStates++;
      if(iter->first == pose_j){
          break;
        }
    }

    StateId lastId;
    StateId id;
    lastId = oldestId;

    double fullDistanceTravelled = 0.0;
    std::vector<double> distances; 
    auto iter = realtimeGraph_.states_.find(pose_i);
    ++iter;
    for(; iter != realtimeGraph_.states_.end(); ++iter) {
      id = iter->first;
      if(id.value() > pose_j.value())
        break;
      double ds = (realtimeGraph_.pose(id).r()-realtimeGraph_.pose(lastId).r()).norm();
      distances.push_back(ds);
      lastId = id;
      fullDistanceTravelled += ds;
    }

    size_t counter = 0; 
    iter = realtimeGraph_.states_.find(pose_i);
    lastId = oldestId;
    ++iter;
    double distanceTravelled = 0.0;
    okvis::kinematics::Transformation T_WS;

    for(; iter != realtimeGraph_.states_.end(); ++iter) {
      id = iter->first;

      if(id.value() <= pose_j.value()){

          distanceTravelled += distances.at(counter);

          Eigen::Vector3d dr = distanceTravelled/fullDistanceTravelled * r_Wold_Wnew;

          T_WS = realtimeGraph_.pose(id);
          okvis::kinematics::Transformation T_WS_set(T_WS.r()+dr,T_WS.q());
          realtimeGraph_.setPose(iter->first, T_WS_set);
          fullGraph_.setPose(iter->first, T_WS_set);
          counter+=1;
        }
      else{

          T_WS = realtimeGraph_.pose(id);
          okvis::kinematics::Transformation T_WS_set(T_WS.r() + r_Wold_Wnew,T_WS.q());
          realtimeGraph_.setPose(iter->first, T_WS_set);
          fullGraph_.setPose(iter->first, T_WS_set);
        }
    }

  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end(); ++iter) {
    Eigen::Vector4d r_Wold_Wnew_hom(r_Wold_Wnew(0) , r_Wold_Wnew(1) , r_Wold_Wnew(2) , 0.0);
    Eigen::Vector4d hPointNew = r_Wold_Wnew_hom + iter->second.hPoint->estimate();
    realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
  }

  return true;
}

bool ViSlamBackend::addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                                  const uint64_t &frame_A_id, const uint64_t frame_B_id,
                                                  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                                  std::vector<float> sensorError, bool isLidar, const std::string robustFunction) {
  if(!isLoopClosing_ && !isLoopClosureAvailable_){
    fullGraph_.addSubmapAlignmentConstraints(submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction);
  }
  else
  {
    addSubmapAlignmentBacklog_.push_back(AddSubmapAlignmentBacklog{submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction});
    LOG(INFO) << "full Graph is being optimized. don't add alignment constraints.";
  }
  return realtimeGraph_.addSubmapAlignmentConstraints(submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction);
}

StateId ViSlamBackend::currentKeyframeStateId(bool considerLoopClosureFrames) const
{
  StateId currentFrame = currentStateId();
  return mostOverlappedStateId(currentFrame, considerLoopClosureFrames);
}

StateId ViSlamBackend::mostOverlappedStateId(StateId frame, bool considerLoopClosureFrames) const
{
  std::set<StateId> allFrames;
  allFrames.insert(keyFrames_.begin(), keyFrames_.end());
  allFrames.insert(imuFrames_.begin(), imuFrames_.end());
  allFrames.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  if(!considerLoopClosureFrames) {
    for(const auto & id : currentLoopClosureFrames_)
      allFrames.erase(id);
  }
  StateId returnId;
  double overlap = 0.0;
  for(auto id : allFrames) {
    if(id==frame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(frame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==frame) {
    OKVIS_THROW(Exception, "most overlapped frame with itself: makes no sense")
  }
  return returnId;
}

StateId ViSlamBackend::currentLoopclosureStateId() const
{
  StateId currentFrame = currentStateId();
  StateId returnId;
  double overlap = 0.0;
  for(auto id : loopClosureFrames_) {
    if(id==currentFrame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(currentFrame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==currentFrame) {
    OKVIS_THROW(Exception, "current frame is loopclosure frame: makes no sense")
  }
  if(overlap>0.5) {
    return returnId;
  }
  return StateId();
}

void ViSlamBackend::clear()
{

  realtimeGraph_.clear();
  fullGraph_.clear();

  multiFrames_.clear();

  auxiliaryStates_.clear(); 
  currentComponentIdx_ = 0; 

  loopClosureFrames_.clear(); 

  imuFrames_.clear(); 
  keyFrames_.clear(); 

  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = false;
  isLoopClosureAvailable_ = false;
  components_.resize(1);

  addStatesBacklog_.clear(); 
  eliminateStates_.clear(); 
  touchedStates_.clear(); 
  touchedLandmarks_.clear(); 

  fullGraphRelativePoseConstraints_.clear(); 

  lastFreeze_ = StateId(); 
}

double ViSlamBackend::overlapFraction(const MultiFramePtr frameA,
                                      const MultiFramePtr frameB) const {

  OKVIS_ASSERT_TRUE(Exception, frameA->numFrames() == frameB->numFrames(),
                    "must be same number of frames")
  const size_t numFrames = frameA->numFrames();
  const MultiFramePtr frames[2] = {frameA, frameB};

  std::set<LandmarkId> landmarks[2];
  std::vector<cv::Mat> detectionsImg[2];
  detectionsImg[0].resize(numFrames);
  detectionsImg[1].resize(numFrames);
  std::vector<cv::Mat> matchesImg[2];
  matchesImg[0].resize(numFrames);
  matchesImg[1].resize(numFrames);

  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      detectionsImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      matchesImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      const size_t num = frames[f]->numKeypoints(im);
      cv::KeyPoint keypoint;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        cv::circle(detectionsImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                   cv::FILLED);
        uint64_t lmId = frames[f]->landmarkId(im, k);
        if (lmId != 0) {
          landmarks[f].insert(LandmarkId(lmId));
        }
      }
    }
  }

  std::set<LandmarkId> matches;
  std::set_intersection(
      landmarks[0].begin(), landmarks[0].end(),landmarks[1].begin(),
      landmarks[1].end(), std::inserter(matches,matches.begin()));

  if(matches.size() == 0) {
    return 0.0;
  }

  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::KeyPoint keypoint;
      const size_t num = frames[f]->numKeypoints(im);
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        if (matches.count(LandmarkId(frames[f]->landmarkId(im, k)))) {
          cv::circle(matchesImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                     cv::FILLED);
        }
      }
    }
  }

  double overlap[2];
  for(size_t f=0; f<2; ++f) {
    int intersectionCount = 0;
    int unionCount = 0;
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::Mat intersectionMask, unionMask;
      cv::bitwise_and(matchesImg[f].at(im), detectionsImg[f].at(im), intersectionMask);
      cv::bitwise_or(matchesImg[f].at(im), detectionsImg[f].at(im), unionMask);
      intersectionCount += cv::countNonZero(intersectionMask);
      unionCount += cv::countNonZero(unionMask);
    }
    overlap[f] = double(intersectionCount)/double(unionCount);
  }

  return std::min(overlap[0], overlap[1]);
}

void ViSlamBackend::writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix)
{
  realtimeGraph_.writeLidarDebugStatisticsCsv(csvFilePrefix);
}

}  
