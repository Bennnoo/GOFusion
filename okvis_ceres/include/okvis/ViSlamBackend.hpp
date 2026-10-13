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
 * @file okvis/ViSlamBackend.hpp
 * @brief Header file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_VISLAMBACKEND_HPP_
#define INCLUDE_OKVIS_VISLAMBACKEND_HPP_

#include <atomic>
#include <functional>
#include <mutex>

#include <okvis/ViGraphEstimator.hpp>
#include "se/map/map.hpp"

#include <Eigen/StdVector>

namespace okvis {

class ViSlamBackend 
{
 public:
  using FinalBaProgressCallback =
      std::function<void(int stage, int iteration, int max_iterations)>;

  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ViSlamBackend() {
    needsFullGraphOptimisation_ = false;
    isLoopClosing_ = false;
    isLoopClosureAvailable_ = false;
    components_.resize(1);
    gpsObservability_ = false;
  }

  virtual ~ViSlamBackend() {}

  bool areLandmarksInFrontOfCameras() const {
    return realtimeGraph_.areLandmarksInFrontOfCameras();
  }

  int addCamera(
      const okvis::CameraParameters & cameraParameters);

  int addImu(const okvis::ImuParameters & imuParameters);

  int addGps(const okvis::GpsParameters & gpsParameters);

  bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque & imuMeasurements,
                 bool asKeyframe);

  void printStates(StateId stateId, std::ostream & buffer) const;

  bool addLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark, bool isInitialised);

  LandmarkId addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised);

  bool setLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark, bool isInitialised);

  bool isObserved(KeypointIdentifier kid) const {
    return realtimeGraph_.observations_.count(kid) != 0;
  }

  template<class GEOMETRY_TYPE>
  bool addObservation(LandmarkId landmarkId, StateId stateId,
                      size_t camIdx, size_t keypointIdx, bool useCauchy = true) {
    KeypointIdentifier kid{stateId.value(), camIdx, keypointIdx};
    OKVIS_ASSERT_TRUE_DBG(Exception, multiFrames_.count(stateId), "frame does noe exist")

    MultiFramePtr multiFrame = multiFrames_.at(stateId);
    const bool success = realtimeGraph_.addObservation<GEOMETRY_TYPE>(
          *multiFrame, landmarkId, kid, useCauchy);
    OKVIS_ASSERT_TRUE_DBG(Exception, !auxiliaryStates_.at(stateId).isPoseGraphFrame,
                      "not allowed to have observations in pose graph frames!")
    if(isLoopClosing_ || isLoopClosureAvailable_) {
      touchedStates_.insert(stateId);
      touchedLandmarks_.insert(landmarkId);
    } else {
      fullGraph_.addObservation<GEOMETRY_TYPE>(*multiFrame, landmarkId, kid, useCauchy);
    }
    return success;
  }

  bool removeObservation(StateId stateId,  size_t camIdx,
                         size_t keypointIdx);

  bool setObservationInformation(StateId stateId, size_t camIdx,
                                 size_t keypointIdx, const Eigen::Matrix2d & information);

  void setDetectorUniformityRadius(double uniformityRadius) {
    kptradius_ = 0.09 * uniformityRadius / 36.0;
  }

  bool applyStrategy(size_t numKeyframes, size_t numLoopClosureFrames, size_t numImuFrames,
                     std::set<StateId> & affectedFrames,
                     bool expand = true);

  void optimiseRealtimeGraph(
      int numIter, std::vector<StateId>& updatedStates,
      int numThreads = 1, bool verbose = false, bool onlyNewestState = false, bool isInitialised = true);

  bool setOptimisationTimeLimit(double timeLimit, int minIterations);

  bool isLandmarkAdded(LandmarkId landmarkId) const;

  bool isLandmarkInitialised(LandmarkId landmarkId) const;

  bool getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const;

  size_t getLandmarks(MapPoints &landmarks) const;

  okvis::MultiFramePtr multiFrame(StateId stateId) const {
    if(!multiFrames_.count(stateId)) {
      return nullptr;
    }
    return multiFrames_.at(stateId);
  }

  const kinematics::TransformationCacheless & pose(StateId id) const {
    return realtimeGraph_.pose(id);
  }

  const SpeedAndBias & speedAndBias(StateId id) const {
    return realtimeGraph_.speedAndBias(id);
  }

  const kinematics::TransformationCacheless & extrinsics(StateId id, uchar camIdx) const {
    return realtimeGraph_.extrinsics(id, camIdx);
  }

  kinematics::Transformation T_GW() const {
    return realtimeGraph_.T_GW();
  }

  void gpsMeasurements(StateId stateId, AlignedVector<Eigen::Vector3d>& gpsMeasurements) {
    return realtimeGraph_.gpsMeasurements(stateId, gpsMeasurements);
  }

  bool isKeyframe(StateId id) const {
    return realtimeGraph_.isKeyframe(id);
  }

  bool isPoseGraphFrame(StateId id) const {

    return auxiliaryStates_.at(id).isPoseGraphFrame;
  }

  bool isPlaceRecognitionFrame(StateId id) const {

    return auxiliaryStates_.at(id).isPlaceRecognitionFrame;
  }

  double trackingQuality(StateId id) const;

  bool setKeyframe(StateId id, bool isKeyframe);

  bool closedLoop(StateId id) const {

    return auxiliaryStates_.at(id).closedLoop;
  }

  const std::set<StateId> & recentLoopClosureFrames(StateId id) const {

    return auxiliaryStates_.at(id).recentLoopClosureFrames;
  }

  StateId currentStateId() const {
    return realtimeGraph_.currentStateId();
  }

  StateId currentKeyframeStateId(bool considerLoopClosureFrames = true) const;

  StateId mostOverlappedStateId(StateId frame, bool considerLoopClosureFrames = true) const;

  StateId currentLoopclosureStateId() const;

  StateId stateIdByAge(size_t age) const {
    return realtimeGraph_.stateIdByAge(age);
  }

  bool setPose(StateId id, const kinematics::TransformationCacheless & pose);

  bool setSpeedAndBias(StateId id, const SpeedAndBias & speedAndBias);

  bool setExtrinsics(
      StateId id, uchar camIdx, const kinematics::TransformationCacheless & extrinsics);

  size_t numFrames() const {
    return multiFrames_.size();
  }

  bool isInImuWindow(StateId id) const;

  okvis::Time timestamp(StateId id) const;

  void drawOverheadImage(cv::Mat & image, int idx=0) const;

  bool getObservedIds(StateId id, std::set<StateId>& observedIds) const;

  bool isLoopClosureFrame(StateId frameId) const;

  bool isRecentLoopClosureFrame(StateId frameId) const;

  int cleanUnobservedLandmarks();

  bool mergeLandmark(const LandmarkId &fromId, const LandmarkId &intoId);

  int mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds);

  bool writeFinalCsvTrajectory(const std::string& csvFileName, bool rpg = false) const;

  bool writeGlobalCsvTrajectory(const std::string& csvFileName) const;

  struct FinalTrajectoryPosition {
    uint64_t state_id = 0;
    uint64_t timestamp_ns = 0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
  };

  bool getFinalTrajectoryPositions(std::vector<FinalTrajectoryPosition>& positions) const;

  bool attemptLoopClosure(StateId pose_i, StateId pose_j,
                          const kinematics::Transformation& T_Si_Sj,
                          const Eigen::Matrix<double, 6, 6>& information,
                          bool & skipFullGraphOptimisation,
                          double driftPercentageHeuristic);

  void addLoopClosureFrame(StateId loopClosureFrameId, std::set<LandmarkId> &loopClosureLandmarks,
                           bool skipFullGraphOptimisation);

  bool needsFullGraphOptimisation() const {return needsFullGraphOptimisation_; }

  void optimiseFullGraph(int numIter, ::ceres::Solver::Summary &summary,
                         int numThreads = 1, bool verbose = false,
                         const FinalBaProgressCallback& progressCallback = {},
                         int progressStage = 0);

  void doFinalBa(int numIter, ::ceres::Solver::Summary &summary,
                 std::set<StateId> & updatedStatesBa,
                 double extrinsicsPositionUncertainty = 0.0,
                 double extrinsicsOrientationUncertainty = 0.0,
                 int numThreads = 1, bool verbose = false,
                 const FinalBaProgressCallback& progressCallback = {});

  bool saveMap(std::string path);

  bool isLoopClosing() const { return isLoopClosing_; }

  bool isLoopClosureAvailable() const { return isLoopClosureAvailable_ && !isLoopClosing_; }

  bool synchroniseRealtimeAndFullGraph(std::vector<StateId>& updatedStates);

  bool isSynched() const {
    return fullGraph_.isSynched(realtimeGraph_);
  }

  double overlapFraction(const MultiFramePtr frameA,
                                const MultiFramePtr frameB) const;

  const std::set<StateId>& keyFrames() const {return keyFrames_;}

  const std::set<StateId>& imuFrames() const {return imuFrames_;}

  const std::set<StateId>& loopClosureFrames() const {return loopClosureFrames_;}

  void eliminateImuFrames(size_t numImuFrames, std::set<StateId> & affectedFrames);

  bool addGpsMeasurementsOnAllGraphs(GpsMeasurementDeque& gpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque);

  bool tryGpsAlignment();

  void dumpGpsResiduals(const std::string &gpsResCsvFileName)
  {

    realtimeGraph_.dumpGpsResiduals(gpsResCsvFileName);
  }

  bool attemptFullGpsAlignment(StateId gpsLossId , StateId gpsReturnId, const okvis::kinematics::Transformation& T_GW_new);

  bool attemptPosGpsAlignment(StateId gpsLossId , StateId gpsReturnId, const Eigen::Vector3d& posAlignVec);

  void addGpsAlignmentFrame(StateId gpsLossFrameId);

  bool addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                     const uint64_t& frame_A_id, const uint64_t frame_B_id,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                     std::vector<float> sensorError, bool isLidar=false, const std::string robustFunction = "Tukey");

  void clear();

  void writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix);

private:

  bool convertToPoseGraphMst(const std::set<StateId> &framesToConvert,
                             const std::set<StateId> &framesToConsider,
                             std::set<StateId> &affectedFrames);

  int expandKeyframe(StateId keyframe);


  std::map<StateId, MultiFramePtr> multiFrames_; 

  struct AuxiliaryState
  {
    bool isKeyframe = false;             
    bool isImuFrame = false;             
    bool isPoseGraphFrame = false;       
    bool closedLoop = false;             
    bool isLost = false;                 
    bool isPlaceRecognitionFrame = true; 
    StateId loopId; 
    std::set<StateId> recentLoopClosureFrames; 
  };
  std::map<StateId, AuxiliaryState> auxiliaryStates_; 

  struct Component0 {
    std::set<StateId> poseIds; 
    StateId referenceId; 
    StateId connectedReferenceId; 
  };
  std::vector<Component0> components_; 
  size_t currentComponentIdx_ = 0; 

  std::set<StateId> loopClosureFrames_; 

  std::set<StateId> imuFrames_; 
  std::set<StateId> keyFrames_; 

  ViGraphEstimator realtimeGraph_; 
  ViGraphEstimator fullGraph_; 

  std::atomic_bool needsFullGraphOptimisation_; 
  std::atomic_bool isLoopClosing_; 
  std::atomic_bool isLoopClosureAvailable_; 
  std::set<StateId> currentLoopClosureFrames_; 

  struct AddStatesBacklog {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Time timestamp; 
    StateId id; 
    ImuMeasurementDeque imuMeasurements; 
  };

  AlignedVector<AddStatesBacklog> addStatesBacklog_; 
  std::map<StateId, StateId> eliminateStates_; 
  std::set<StateId> touchedStates_; 
  std::set<LandmarkId> touchedLandmarks_; 

  std::set<StateId> updatedStatesLoopClosureAttempt_; 

  struct RelPoseInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    kinematics::Transformation T_Si_Sj; 
    Eigen::Matrix<double, 6, 6> information; 
    StateId pose_i; 
    StateId pose_j; 
  };

  AlignedVector<RelPoseInfo> fullGraphRelativePoseConstraints_; 

  StateId lastFreeze_; 

  double kptradius_ = 0.09; 

  std::atomic_bool gpsObservability_; 
  struct AddGpsBacklog{
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      StateId id;
      GpsMeasurement gpsMeasurement;
      ImuMeasurementDeque imuMeasurements;
      bool reInitFlag;
  };
  AlignedVector<AddGpsBacklog> addGpsBacklog_;

  struct AddSubmapAlignmentBacklog{
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      const SupereightMapType* submap_ptr;
      uint64_t frame_A_id;
      uint64_t frame_B_id;
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud;
      std::vector<float> sensorError; 
      bool isLidar;
      std::string robustFunction;
  };
  AlignedVector<AddSubmapAlignmentBacklog> addSubmapAlignmentBacklog_;

  public: 
  AlignedMap<StateId, AlignedMap<uint64_t, kinematics::Transformation>> T_AiS_; 
};

}  

#endif 
