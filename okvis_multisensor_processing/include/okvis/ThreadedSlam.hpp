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
 * @file ThreadedSlam.hpp
 * @brief Header file for the ThreadedSlam3 class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_THREADEDSLAM3_HPP_
#define INCLUDE_OKVIS_THREADEDSLAM3_HPP_

#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>

#include <opencv2/core/core.hpp>

#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/TrackingImageDownsampler.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ViVisualizer.hpp>
#include <okvis/timing/Timer.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/ViInterface.hpp>

#include <se/supereight.hpp>
#include <okvis/config_mapping.hpp>
#include <okvis/mapTypedefs.hpp>
#include <okvis/SubmappingInterface.hpp>
#include <Eigen/StdVector>

namespace okvis {

class ThreadedSlam : public ViInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  ThreadedSlam(okvis::ViParameters& parameters, std::string dBowDir, se::SubMapConfig submapConfig = se::SubMapConfig());

  virtual ~ThreadedSlam() override;

  virtual bool addImages(const okvis::Time & stamp,
                         const std::map<size_t, cv::Mat> & images,
                         const std::map<size_t, cv::Mat> & depthImages
                         = std::map<size_t, cv::Mat>()) override final;

  bool buildStereoPreview(const okvis::Time& stamp,
                          const std::map<size_t, cv::Mat>& images,
                          std::map<std::string, cv::Mat>& output_images);

  virtual bool addImuMeasurement(const okvis::Time & stamp,
                                 const Eigen::Vector3d & alpha,
                                 const Eigen::Vector3d & omega) override final;

  virtual bool addLidarMeasurement(const okvis::Time & stamp,
                                   const Eigen::Vector3d & rayMeasurement) override final;

  bool addDepthMeasurement(const okvis::Time &stamp,
                           const cv::Mat &depthImage) override final;

  bool addTaggedDepthMeasurement(const CameraMeasurement& measurement);

  virtual bool addGpsMeasurement(const okvis::Time & stamp,
                                 const Eigen::Vector3d & posGps,
                                 const Eigen::Vector3d & errGps);

  virtual bool addGeodeticGpsMeasurement(const okvis::Time & stamp,
                                         double lat, double lon, double height,
                                         double hAcc, double vAcc);

    bool addSubmapAlignmentConstraints(const SupereightMapType* submap_A_ptr,
                                       const SupereightMapType* submap_B_ptr,
                                       const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                                       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                       std::vector<float> sensorError);

  virtual void setBlocking(bool blocking) override final;

  void enqueueGsOptimizedPoses(
      const std::map<uint64_t, Eigen::Matrix4f>& poses_by_state_id);

  bool processFrame();

  float latestFrontendTrackingFps() const {
    return frontendTrackingFpsHz_.load(std::memory_order_relaxed);
  }

  void display(std::map<std::string, cv::Mat> & images) override final;

  void setFinalTrajectoryCsvFile(const std::string & finalTrajectoryCsvFileName, bool rpg = false) {
    finalTrajectoryCsvFileName_ = finalTrajectoryCsvFileName;
    rpg_ = rpg;
  }

  void setMapCsvFile(const std::string & mapCsvFileName) {
    mapCsvFileName_ = mapCsvFileName;
  }

  void stopThreading();

  void writeFinalTrajectoryCsv();

  void writeGlobalTrajectoryCsv(const std::string& csvFileName);

  void writeDebugStatisticsCsv(const std::string& csvFilePrefix);

  void doFinalBa();

  void setFinalBaProgressCallback(
      const ViSlamBackend::FinalBaProgressCallback& callback)
  {
    finalBaProgressCallback_ = callback;
  }

  bool getFinalTrajectoryPositions(std::map<uint64_t, Eigen::Vector3f>& positionsById) const;

  bool getFinalTrajectoryPoses(
      std::vector<ViSlamBackend::FinalTrajectoryPosition>& poses) const;

  bool saveMap();

  bool exportSe2Map();

  bool createFinalSe2Map();

  void dumpGpsResiduals(const std::string & gpsResCsvFileName);

  bool needsNewLidarKeyframe();

private:
  bool enqueueDepthMeasurement(CameraMeasurement measurement);

  void startThreads();

  void init();

  bool getNextFrame(MultiFramePtr &multiFrame);

  void optimisePublishMarginalise(MultiFramePtr multiFrame, const Eigen::Vector3d& gyroReading);

  void visualisationLoop();

  void publishingLoop();

  void computeLiveDepthMeasurements(const kinematics::Transformation& T_WS_live, const okvis::Time& curTime,
                                    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints, std::vector<float>& observedSigmas);

  std::atomic_bool blocking_; 

  threadsafe::Queue<std::vector<okvis::CameraMeasurement> > cameraMeasurementsReceived_;

  threadsafe::Queue<okvis::ImuMeasurement, Eigen::aligned_allocator<okvis::ImuMeasurement>>
      imuMeasurementsReceived_;

  threadsafe::Queue<okvis::LidarMeasurement , Eigen::aligned_allocator<okvis::LidarMeasurement>>
      lidarMeasurementsReceived_;

  threadsafe::Queue<okvis::CameraMeasurement, Eigen::aligned_allocator<okvis::CameraMeasurement>>
      depthMeasurementsReceived_;

  struct AlignmentTerm{
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      const SupereightMapType* submap_A_ptr; 
      const SupereightMapType* submap_B_ptr; 
      uint64_t frame_A_id; 
      uint64_t frame_B_id; 
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud_B; 
      std::vector<float> sensorError; 

      AlignmentTerm()
      {};

      AlignmentTerm(const SupereightMapType* submap_A_ptr,
                    const SupereightMapType* submap_B_ptr,
                    const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud_B,
                    std::vector<float> sensorError)
              :  submap_A_ptr(submap_A_ptr), submap_B_ptr(submap_B_ptr),
              frame_A_id(frame_A_id), frame_B_id(frame_B_id), pointCloud_B(pointCloud_B), sensorError(sensorError)
      {
      };
  };
  threadsafe::Queue<AlignmentTerm, Eigen::aligned_allocator<AlignmentTerm>>
      submapAlignmentFactorsReceived_;

  threadsafe::Queue<okvis::GpsMeasurement> gpsMeasurementsReceived_;

  threadsafe::Queue<ViVisualizer::VisualizationData::Ptr> visualisationData_;

  threadsafe::Queue<cv::Mat> overheadImages_;

  threadsafe::Queue<std::vector<cv::Mat>> visualisationImages_;

  ViVisualizer visualizer_; 

  std::thread visualisationThread_; 
  std::thread publishingThread_; 

  std::thread optimisationThread_; 
  std::thread fullGraphOptimisationThread_; 

  struct PublicationData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    State state; 
    TrackingState trackingState; 
    std::shared_ptr<AlignedMap<StateId, State>> updatedStates; 
    std::shared_ptr<MapPointVector> landmarksPublish; 
  };
  threadsafe::Queue<PublicationData> lastOptimisedQueue_; 
  threadsafe::Queue<PublicationData> publicationQueue_; 
  std::set<StateId> affectedStates_; 

  std::atomic_bool hasStarted_; 

  bool firstFrame_ = true; 
  ImuMeasurementDeque imuMeasurementDeque_;  
  GpsMeasurementDeque gpsMeasurementDeque_;  
  LidarMeasurementDeque lidarMeasurementDeque_;  
  DepthMeasurementDeque depthMeasurementDeque_; 

  AlignedMap<StateId, ImuMeasurementDeque> imuMeasurementsByFrame_;

  okvis::ViSlamBackend estimator_;    
  okvis::Frontend frontend_;      
  okvis::Trajectory trajectory_;
  std::ofstream trajectory_debug_output_;
  std::ofstream base_frame_debug_output_;

  okvis::ViParameters parameters_; 
  TrackingImageDownsampler trackingImageDownsampler_;

  std::string finalTrajectoryCsvFileName_; 
  bool rpg_ = false; 
  std::string mapCsvFileName_; 

  ::ceres::Solver::Summary posegraphOptimisationSummary_; 

  State lastOptimisedState_; 
  State preLastOptimisedState_; 

  std::atomic_bool shutdown_; 

  se::SubMapConfig submapConfig_; 
  okvis::kinematics::Transformation T_SD_; 
  bool useAlignmentFactors_;
  const SupereightMapType* previousSubmap_ = nullptr; 
  uint64_t previousSubmapId_ = -1;
  size_t previouslyObservedLidarPoints_ = 0;
  size_t noOverlapCounter_ = 0;
  std::set<StateId> lidarKeyframes_;

  ViSlamBackend::FinalBaProgressCallback finalBaProgressCallback_;


  int sensorMeasurementDownsamplingCounter_ = 0;
  std::shared_ptr<se::PinholeCamera> depthCamera_;
  const size_t cameraInputQueueSize_;
  const size_t depthMeasurementQueueSize_;
  std::atomic<float> frontendTrackingFpsHz_{0.0f};

  AlignedMap<uint64_t, Eigen::Matrix4f> pendingGsPoseUpdates_;
  std::mutex pendingGsPoseUpdatesMtx_;

  void applyPendingGsPoseUpdates();

  void saveAlignedPoints(size_t id,
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points,
    std::string type);
};

}  

#endif 
