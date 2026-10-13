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
 * @file okvis/ViGraph.hpp
 * @brief Header file for the ViGraph2 class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_VIGRAPH2_HPP_
#define INCLUDE_OKVIS_VIGRAPH2_HPP_

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

#include <okvis/kinematics/Transformation.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/mapTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/TwoPoseGraphError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/DepthError.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>
#include <okvis/ceres/GpsErrorAsynchronous.hpp>
#include <okvis/ceres/SubmapIcpError.hpp>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>

#include <se/supereight.hpp>
#include <Eigen/StdVector>

namespace okvis {

  enum gpsStatus{
      Off = 0,
      Idle = 1,
      Initialising = 2,
      Initialised = 3,
      ReInitialising = 4
  };

class ViGraph
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  friend class ViSlamBackend;
  friend class Component;

  ViGraph();

  ~ViGraph() { problem_.reset(); }

  int addCamera(const okvis::CameraParameters & cameraParameters);

  int addImu(const okvis::ImuParameters & imuParameters);

  int addGps(const okvis::GpsParameters & gpsParameters);

  StateId addStatesInitialise(const Time& timestamp, const ImuMeasurementDeque & imuMeasurements,
                              const cameras::NCameraSystem & nCameraSystem);

  StateId addStatesPropagate(const Time& timestamp, const ImuMeasurementDeque & imuMeasurements,
                             bool isKeyframe);

  bool addStatesFromOther(StateId stateId, const ViGraph & other);

  bool addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &homogeneousPoint, bool initialised);

  LandmarkId addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised);

  bool removeLandmark(LandmarkId landmarkId);

  bool setLandmarkInitialised(LandmarkId landmarkId, bool initialised);

  bool setLandmarkQuality(LandmarkId landmarkId, double quality) {

    landmarks_.at(landmarkId).quality = quality;
    return true;
  }

  bool isLandmarkInitialised(LandmarkId landmarkId) const;

  bool isLandmarkAdded(LandmarkId landmarkId) const;

  const Eigen::Vector4d & landmark(LandmarkId id) const;

  bool getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const;

  size_t getLandmarks(MapPoints & landmarks) const;

  bool landmarkExists(LandmarkId landmarkId) const;

  bool setLandmark(LandmarkId id, const Eigen::Vector4d & homogeneousPoint, bool initialised);

  bool setLandmark(LandmarkId id, const Eigen::Vector4d & homogeneousPoint);

  bool addOneSidedDepthError(KeypointIdentifier keypointId)
  {
    Observation &obs = observations_.at(keypointId);
    if (obs.depthError.residualBlockId) {
      return false;
    }
    State &state = states_.at(StateId(keypointId.frameId));
    obs.depthError.errorTerm.reset(new ceres::OneSidedDepthError(0.1, 0.001));
    obs.depthError.residualBlockId
      = problem_->AddResidualBlock(obs.depthError.errorTerm.get(),
                                   nullptr,
                                   state.pose->parameters(),
                                   landmarks_.at(obs.landmarkId).hPoint->parameters(),
                                   state.extrinsics.at(keypointId.cameraIndex)->parameters());

    landmarks_.at(obs.landmarkId).observations.at(keypointId) = obs;
    state.observations.at(keypointId) = obs;

    return true;
  }

  bool removeOneSidedDepthError(KeypointIdentifier keypointId)
  {
    Observation &obs = observations_.at(keypointId);
    if (!obs.depthError.residualBlockId) {
      return false;
    }
    State &state = states_.at(StateId(keypointId.frameId));
    problem_->RemoveResidualBlock(obs.depthError.residualBlockId);
    obs.depthError.errorTerm.reset();
    obs.depthError.residualBlockId = nullptr;

    landmarks_.at(obs.landmarkId).observations.at(keypointId) = obs;
    state.observations.at(keypointId) = obs;

    return true;
  }

  bool areLandmarksInFrontOfCameras() const;

  template<class GEOMETRY_TYPE>
  bool addObservation(const MultiFrame& multiFrame, LandmarkId landmarkId,
                      KeypointIdentifier keypointId, bool useCauchy = true) {
    OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark not added")

    OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.at(landmarkId).observations.count(keypointId) == 0,
                          "observation already exists")
    OKVIS_ASSERT_TRUE_DBG(Exception, observations_.count(keypointId) == 0,
                          "observation already exists")
    OKVIS_ASSERT_TRUE_DBG(Exception, multiFrame.landmarkId(
                            keypointId.cameraIndex, keypointId.keypointIndex) == landmarkId.value(),
                          "observation already exists")

    Eigen::Vector2d measurement;
    multiFrame.getKeypoint(keypointId.cameraIndex, keypointId.keypointIndex, measurement);
    Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
    double size = 1.0;
    multiFrame.getKeypointSize(keypointId.cameraIndex, keypointId.keypointIndex, size);
    information *= 64.0 / (size * size);

    Observation observation;
    observation.errorTerm.reset(new ceres::ReprojectionError<GEOMETRY_TYPE>(
                    multiFrame.template geometryAs<GEOMETRY_TYPE>(keypointId.cameraIndex),
                    keypointId.cameraIndex, measurement, information));

    State& state = states_.at(StateId(keypointId.frameId));
    observation.residualBlockId = problem_->AddResidualBlock(
        observation.errorTerm.get(),
        useCauchy&&cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : nullptr,
        state.pose->parameters(), landmarks_.at(landmarkId).hPoint->parameters(),
        state.extrinsics.at(keypointId.cameraIndex)->parameters());
    observation.landmarkId = landmarkId;

    observations_[keypointId] = observation;
    landmarks_.at(landmarkId).observations[keypointId] = observation;
    state.observations[keypointId] = observation;

    covisibilitiesComputed_ = false;

    return true;
  }

  bool addExternalObservation(
      const std::shared_ptr<const ceres::ReprojectionError2dBase> & reprojectionError,
       LandmarkId landmarkId, KeypointIdentifier keypointId, bool useCauchy = true) {
    OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark not added")

    OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.at(landmarkId).observations.count(keypointId) == 0,
                          "observation already exists")
    OKVIS_ASSERT_TRUE_DBG(Exception, observations_.count(keypointId) == 0,
                          "observation already exists")

    Observation observation;
    observation.errorTerm = reprojectionError->clone();

    State& state = states_.at(StateId(keypointId.frameId));
    observation.residualBlockId = problem_->AddResidualBlock(
        observation.errorTerm.get(),
        useCauchy&&cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : nullptr,
        state.pose->parameters(), landmarks_.at(landmarkId).hPoint->parameters(),
        state.extrinsics.at(keypointId.cameraIndex)->parameters());
    observation.landmarkId = landmarkId;

    observations_[keypointId] = observation;
    landmarks_.at(landmarkId).observations[keypointId] = observation;
    state.observations[keypointId] = observation;

    covisibilitiesComputed_ = false;

    return true;
  }

  bool removeObservation(KeypointIdentifier keypointId);

  bool computeCovisibilities();

  int covisibilities(StateId pose_i, StateId pose_j) const;

  bool addRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                 const kinematics::Transformation &T_S0S1,
                                 const Eigen::Matrix<double, 6, 6>& information);

  bool removeRelativePoseConstraint(StateId poseId0, StateId poseId1);

   bool addGpsMeasurement(StateId poseId, GpsMeasurement &gpsMeas, const ImuMeasurementDeque &imuMeasurements);

  int checkValidGpsMeasurements(GpsMeasurementDeque& inputGpsMeasurementDeque, GpsMeasurementDeque&gpsMeasurementDeque);

  bool addGpsMeasurements(GpsMeasurementDeque& gpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque, std::deque<StateId>* sids);

  void gpsMeasurements(StateId stateId, AlignedVector<Eigen::Vector3d>& gpsMeasurements);

  void freezeGpsExtrinsics();

  void unfreezeGpsExtrinsics();

  gpsStatus getGpsStatus(){return gpsStatus_;};

  void setGpsStatus(gpsStatus status){gpsStatus_ = status;};

  bool findStateId(StateId sid){ return states_.count(sid);}

  bool isGpsObservable(){return gpsObservability_;}

  bool isGpsFixed(){return gpsFixed_;}

  bool needsFullGpsAlignment(StateId& gpsLossId, StateId& posAlignId, okvis::kinematics::Transformation& T_GW_new);

  bool needsPosGpsAlignment(StateId& gpsLossId, StateId& gpsAlignId, Eigen::Vector3d& posError);

  bool needsInitialGpsAlignment();

  bool needsGpsReInit();

  void reInitGpsExtrinsics();

  bool initializationStrategy(kinematics::Transformation& T_GW);

  void addGpsInitFactors();

  void resetFullGpsAlignment();

  void resetPosGpsAlignment();

  void resetInitialGpsAlignment(){
    gpsInitMap_.clear();
    gpsInitImuQueue_.clear();
    needsInitialAlignment_=false;
  }

  bool setGpsExtrinsics(const kinematics::TransformationCacheless & T_GW);

  void dumpGpsResiduals(const std::string & gpsResCsvFileName){

      std::ofstream residualsOutput(gpsResCsvFileName);
      residualsOutput << "# tk , tg , res_x , res_y , res_z \n";

      std::vector<::ceres::ResidualBlockId> residualIds;
      problem_->GetResidualBlocksForParameterBlock(states_.at(okvis::StateId(1)).T_GW->parameters(), &residualIds);

      std::cout << "[DEBUG INFO Summary] Dumping " << residualIds.size() << " residuals for " << states_.size() << " states." << std::endl;

      for(auto res :residualIds){
          double cost = 0.0;

          const ::ceres::CostFunction* costFct;
          costFct = problem_->GetCostFunctionForResidualBlock(res);

          static_cast<okvis::ceres::GpsErrorAsynchronous*>(const_cast<::ceres::CostFunction*>(costFct))->setInformation(Eigen::Matrix3d::Identity());

          okvis::Time tg;
          tg = static_cast<okvis::ceres::GpsErrorAsynchronous*>(const_cast<::ceres::CostFunction*>(costFct))->tg();
          okvis::Time tk;
          tk = static_cast<okvis::ceres::GpsErrorAsynchronous*>(const_cast<::ceres::CostFunction*>(costFct))->tk();

          Eigen::Matrix<double,3,1> residuals;
          problem_->EvaluateResidualBlock(res,false,&cost,residuals.data(),nullptr);

          residualsOutput << tk << " , " << tg << " , "
                          << residuals(0) << " , "
                          << residuals(1) << " , "
                          << residuals(2) << std::endl;

      }
      residualsOutput.close();

  }

  void getLandmarkHessian(LandmarkId lmId, Eigen::Matrix3d& H);

  bool checkForGpsInit(okvis::kinematics::Transformation& T_GW, std::set<StateId> consideredStates, double* yaw_error = nullptr);

  bool addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                     const uint64_t& frame_A_id, const uint64_t frame_B_id,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                     std::vector<float> sensorError, bool isLidar = false, const std::string robustFunction = "Tukey");

  const kinematics::TransformationCacheless & pose(StateId id) const;

  const SpeedAndBias & speedAndBias(StateId id) const;

  const kinematics::TransformationCacheless & extrinsics(StateId id, uchar camIdx) const;

  const kinematics::TransformationCacheless & T_GW() const;

  const kinematics::TransformationCacheless & T_GW(StateId id) const;

  bool isKeyframe(StateId id) const;

  Time timestamp(StateId id) const;

  StateId currentStateId() const;

  StateId stateIdByAge(size_t age) const;

  bool setKeyframe(StateId id, bool isKeyframe);

  bool setPose(StateId id, const kinematics::TransformationCacheless & pose);

  bool setSpeedAndBias(StateId id, const SpeedAndBias & speedAndBias);

  bool setExtrinsics(
      StateId id, uchar camIdx, const kinematics::TransformationCacheless & extrinsics) const;

  bool setExtrinsicsVariable();

  bool softConstrainExtrinsics(double posStd, double rotStd);

  const ::ceres::Solver::Options & options() const { return options_; }

  ::ceres::Solver::Options & options() { return options_; }

  const ::ceres::Solver::Summary & summary() const { return summary_; }

  void optimise(int maxIterations, int numThreads, bool verbose);

  bool setOptimisationTimeLimit(double timeLimit, int minIterations);

  int cleanUnobservedLandmarks(
      std::map<LandmarkId, std::set<KeypointIdentifier>> *removed = nullptr);

  void updateLandmarks();

  void writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix);

protected:

  void checkObservations() const;

  template<typename ErrorTermT>
  struct GraphEdge {
    ::ceres::ResidualBlockId residualBlockId = nullptr; 
    std::shared_ptr<ErrorTermT> errorTerm; 
  };

  template<typename ErrorTermT>
  struct TwoStateGraphEdge : public GraphEdge<ErrorTermT>{
    StateId state0; 
    StateId state1; 
  };

  struct Observation : public GraphEdge<ceres::ReprojectionError2dBase> {
    LandmarkId landmarkId; 
    GraphEdge<ceres::OneSidedDepthError> depthError; 
  };

  using ImuLink = GraphEdge<ceres::ImuErrorBase>;

  using ExtrinsicsLink = GraphEdge<ceres::RelativePoseError>;

  using PosePrior = GraphEdge<ceres::PoseError>;

  using SpeedAndBiasPrior = GraphEdge<ceres::SpeedAndBiasError>;

  using ExtrinsicsPrior = GraphEdge<ceres::PoseError>;

  using TwoPoseLink = TwoStateGraphEdge<ceres::TwoPoseGraphError>;

  using TwoPoseConstLink = TwoStateGraphEdge<ceres::TwoPoseGraphErrorConst>;

  using RelativePoseLink = TwoStateGraphEdge<ceres::RelativePoseError>;

  using GpsFactor = GraphEdge<ceres::GpsErrorAsynchronous>;

  using SubmapAlignmentFactor = GraphEdge<ceres::SubmapIcpError>;

  struct State {

    std::shared_ptr<ceres::PoseParameterBlock> pose; 
    std::shared_ptr<ceres::SpeedAndBiasParameterBlock> speedAndBias; 
    std::vector<std::shared_ptr<ceres::PoseParameterBlock>> extrinsics; 
    std::shared_ptr<ceres::PoseParameterBlock> T_GW; 

    std::map<KeypointIdentifier, Observation> observations; 
    ImuLink nextImuLink; 
    ImuLink previousImuLink; 
    std::vector<ExtrinsicsLink> nextExtrinsicsLink; 
    std::vector<ExtrinsicsLink> previousExtrinsicsLink; 
    PosePrior posePrior; 
    SpeedAndBiasPrior speedAndBiasPrior; 
    std::vector<ExtrinsicsPrior> extrinsicsPriors; 
    std::map<StateId, TwoPoseLink> twoPoseLinks; 
    std::map<StateId, TwoPoseConstLink> twoPoseConstLinks; 
    std::map<StateId, RelativePoseLink> relativePoseLinks; 
    std::vector<GpsFactor> GpsFactors; 

    std::vector<::ceres::ResidualBlockId> mapResIds;
    std::vector<SubmapAlignmentFactor> submapReferenceLinks;
    std::vector<SubmapAlignmentFactor> submapLinks;
    size_t gpsMode; 

    bool isKeyframe = false; 
    okvis::Time timestamp = okvis::Time(0.0); 
  };

  struct Landmark {
    std::shared_ptr<ceres::HomogeneousPointParameterBlock> hPoint; 
    std::map<KeypointIdentifier, Observation> observations; 
    double quality = 0.0; 
  };

  std::vector<okvis::CameraParameters,
      Eigen::aligned_allocator<okvis::CameraParameters> > cameraParametersVec_; 
  std::vector<okvis::ImuParameters,
      Eigen::aligned_allocator<okvis::ImuParameters> > imuParametersVec_; 
  std::vector<okvis::GpsParameters,
      Eigen::aligned_allocator<okvis::GpsParameters> > gpsParametersVec_; 

  std::map<StateId, State> states_; 
  std::map<LandmarkId, Landmark> landmarks_; 
  std::map<KeypointIdentifier, Observation> observations_; 

  gpsStatus gpsStatus_ = gpsStatus::Off; 
  bool gpsObservability_ = false; 
  bool gpsFixed_ = false; 
  kinematics::Transformation T_GW_init_;

  std::set<StateId> gpsStates_; 
  std::multimap<StateId, GpsMeasurement> gpsInitMap_;
  ImuMeasurementDeque gpsInitImuQueue_; 

  bool needsInitialAlignment_ = false; 
  bool needsPositionAlignment_ = false; 
  bool needsFullAlignment_ = false; 

  StateId gpsDropoutId_; 
  StateId positionAlignedId_; 

  std::set<StateId> gpsReInitStates_; 
  bool gpsReInitialised_ = false; 

  okvis::ceres::PoseManifold4d gpsExtrinsicLocalParametrisation_;

  okvis::ceres::HomogeneousPointManifold homogeneousPointManifold_;

  okvis::ceres::PoseManifold poseManifold_;

  std::shared_ptr< ::ceres::Problem> problem_;

  ::ceres::Solver::Options options_;

  ::ceres::Solver::Summary summary_;

  std::shared_ptr< ::ceres::LossFunction> cauchyLossFunctionPtr_; 
  std::shared_ptr< ::ceres::LossFunction> cauchyGpsLossFunctionPtr_; 
  std::shared_ptr< ::ceres::LossFunction> huberLossFunctionPtr_; 
  std::shared_ptr< ::ceres::LossFunction> tukeyDepthLossFunctionPtr_; 
  std::shared_ptr< ::ceres::LossFunction> tukeyLidarLossFunctionPtr_; 

  std::unique_ptr<ceres::CeresIterationCallback> ceresCallback_; 

  std::map<uint64_t, std::map<uint64_t, int>> coObservationCounts_; 

  bool covisibilitiesComputed_ = true;
  std::set<StateId> visibleFrames_; 

  struct AnyState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateId keyframeId; 
    Time timestamp; 
    kinematics::Transformation T_Sk_S; 
    Eigen::Vector3d v_Sk; 
  };
  AlignedMap<StateId, AnyState> anyState_; 

  const GeographicLib::Geocentric& earth_ = GeographicLib::Geocentric::WGS84();
  GeographicLib::LocalCartesian globCartesianFrame_;

  struct lidarDbgInfo {
    uint64_t frameId;
    uint64_t referenceId;

    size_t n_residuals;

    size_t n_nonzro_occ;
    size_t n_non_zero_grad;

    double mean_res;
    double max_residual;
    size_t n_big_residuals; 

    float meanFieldVal;
    float minFieldVal;
    float maxFieldVal;
    float meanGradNorm;
    float maxGradNorm;
  };
  std::vector<lidarDbgInfo> lidarDbgInfos_;
  bool debugLidarResiduals_ = false;

};

}  

#endif 
