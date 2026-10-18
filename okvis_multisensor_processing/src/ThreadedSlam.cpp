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
 * @file ThreadedSlam.cpp
 * @brief Source file for the ThreadedSlam3 class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <chrono>
#include <cmath>
#include <map>

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/ThreadedSlam.hpp>
#include <okvis/DepthUtils.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/LidarMotionUndistortion.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

namespace okvis
{

static const double imuTemporalOverlap = 0.02;

namespace {

std::string cameraDisplayName(const ViParameters& parameters,
                              const size_t camera_index)
{
  return (parameters.nCameraSystem.cameraType(camera_index).isColour
              ? "rgb"
              : "cam")
         + std::to_string(camera_index);
}

}  

ThreadedSlam::ThreadedSlam(ViParameters &parameters, std::string dBowDir, se::SubMapConfig subMapConfig) :
  visualizer_(parameters),
  hasStarted_(false),
  frontend_(parameters.trackingCameraSystem().numCameras(), dBowDir),
  parameters_(parameters),
  trackingImageDownsampler_(parameters_),
  submapConfig_(subMapConfig),
  useAlignmentFactors_(submapConfig_.useMap2LiveFactors),
  cameraInputQueueSize_(
      subMapConfig.cameraInputQueueSize > 0
          ? subMapConfig.cameraInputQueueSize
          : (subMapConfig.realtimePipelineEnabled ? size_t(1) : size_t(2))),
  depthMeasurementQueueSize_(
      subMapConfig.depthMeasurementQueueSize > 0
          ? subMapConfig.depthMeasurementQueueSize
          : (subMapConfig.realtimePipelineEnabled ? size_t(1) : size_t(4)))
{
  setBlocking(false);
  init();

}

void ThreadedSlam::init()
{
  assert(parameters_.nCameraSystem.numCameras() > 0);
  const size_t numCameras = parameters_.nCameraSystem.numCameras();
  shutdown_ = false;

  frontend_.setBriskDetectionOctaves(size_t(parameters_.frontend.octaves));
  frontend_.setBriskDetectionThreshold(parameters_.frontend.detection_threshold);
  frontend_.setBriskDetectionAbsoluteThreshold(parameters_.frontend.absolute_threshold);
  frontend_.setBriskMatchingThreshold(parameters_.frontend.matching_threshold);
  frontend_.setBriskDetectionMaximumKeypoints(size_t(parameters_.frontend.max_num_keypoints));
  frontend_.setKeyframeInsertionOverlapThreshold(float(parameters_.frontend.keyframe_overlap));

  estimator_.addImu(parameters_.imu);
  for (size_t im = 0; im < numCameras; ++im) {

    estimator_.addCamera(parameters_.camera);
  }
  if(parameters_.gps){
    estimator_.addGps(*parameters_.gps);
  }
  estimator_.setDetectorUniformityRadius(parameters_.frontend.detection_threshold);

  if(parameters_.estimator.enforce_realtime) {
    estimator_.setOptimisationTimeLimit(
          parameters_.estimator.realtime_time_limit,
          parameters_.estimator.realtime_min_iterations);
  }

  if(parameters_.output.enable_submapping) {
    if(parameters_.lidar){
      T_SD_ = parameters_.lidar.value().T_SL;
    }
    else{
      std::shared_ptr<const cameras::CameraBase> camera;
      kinematics::Transformation T_SD;
      bool mapping_camera_found = false;
      for(size_t i = 0; i < parameters_.nCameraSystem.numCameras(); i++){
        if(parameters_.nCameraSystem.cameraType(i).isUsedMapping
           && (!parameters_.mappingInputPreprocessor
               || !parameters_.mappingInputPreprocessor->enabled()
               || parameters_.mappingInputPreprocessor->targetCameraIndex() == i)) {
          mapping_camera_found = true;
          const bool mapping_input_preprocessed =
              parameters_.mappingInputEnabledFor(i);
          camera = parameters_.mappingCameraGeometry(
              i, mapping_input_preprocessed
                     ? static_cast<size_t>(submapConfig_.depthImageResDownsampling)
                     : size_t(1));
          T_SD = *parameters_.mappingT_SC(i);
          break;
        }
      }
      if(!mapping_camera_found){
        LOG(WARNING) << "Neither LiDAR nor camera for mapping specified! Running in VI mode without depth." \
        "If you want to run without depth specify enabling_submapping: false in the okvis_config. If you want to use" \
        "the LiDAR, please specify a lidar section in the okvis_config. If you wanted to use depth from cameras " \
        ", please specify mapping: true for the camera from which depth will be integrated.";
        useAlignmentFactors_ = false;
      } else {
        se::PinholeCamera::Config depthCameraConfig;
        Eigen::VectorXd intrinsics;
        camera->getIntrinsics(intrinsics);
        depthCameraConfig.fx = intrinsics(0);
        depthCameraConfig.fy = intrinsics(1);
        depthCameraConfig.cx = intrinsics(2);
        depthCameraConfig.cy = intrinsics(3);
        depthCameraConfig.T_BS = T_SD.T().cast<float>();
        depthCameraConfig.width = camera->imageWidth();
        depthCameraConfig.height = camera->imageHeight();
        depthCameraConfig.near_plane = submapConfig_.near_plane;
        depthCameraConfig.far_plane = submapConfig_.far_plane;
        const bool mapping_input_preprocessed =
            parameters_.mappingInputPreprocessor
            && parameters_.mappingInputPreprocessor->enabled();
        if (mapping_input_preprocessed) {
          depthCamera_.reset(new se::PinholeCamera(depthCameraConfig));
        } else {
          (void) validatedDepthDownsampledSize(
              depthCameraConfig.width,
              depthCameraConfig.height,
              submapConfig_.depthImageResDownsampling,
              "map-to-live depth camera");
          depthCamera_.reset(new se::PinholeCamera(
              depthCameraConfig, submapConfig_.depthImageResDownsampling));
        }
        T_SD_ = T_SD;
      }
    }
  } else if(useAlignmentFactors_){
    useAlignmentFactors_ = false;
    LOG(WARNING) << "Alignment factors have been set in the se2_config but the enable_submapping in the okvis_config is set to false." \
    " Due to this inconsistency you will be running VI mode no submap constraints.";
  }

  if(useAlignmentFactors_ && parameters_.output.enable_submapping){
    LOG(INFO) << "[Info SLAM] SLAM will be using Submap Alignment Constraints.";
  }

  startThreads();
}

void ThreadedSlam::startThreads()
{

  if(parameters_.output.display_matches) {
    visualisationThread_ = std::thread(&ThreadedSlam::visualisationLoop, this);
  }

  publishingThread_ = std::thread(&ThreadedSlam::publishingLoop, this);

}

ThreadedSlam::~ThreadedSlam()
{

  stopThreading();
}

bool ThreadedSlam::addImages(const okvis::Time & stamp,
                             const std::map<size_t, cv::Mat> & images,
                             const std::map<size_t, cv::Mat> & depthImages)
{

  const Time stampCorrected = stamp - Duration(parameters_.camera.image_delay);

  const size_t numCameras = parameters_.nCameraSystem.numCameras();
  std::vector<okvis::CameraMeasurement> frames(numCameras);
  frames.at(0).timeStamp = stampCorrected; 

  bool useFrame = false;
  for(const auto & image : images) {
    if(parameters_.nCameraSystem.isCameraConfigured(image.first) && 
       parameters_.nCameraSystem.cameraType(image.first).isUsed) {
      cv::Mat downsampled;
      const cv::Mat* tracking_image = &image.second;
      if (trackingImageDownsampler_.enabled()) {
        downsampled = trackingImageDownsampler_.process(
            image.first, image.second);
        tracking_image = &downsampled;
      }
      if(tracking_image->channels()==1) {
        frames.at(image.first).measurement.image = *tracking_image;
      } else {
        cv::cvtColor(*tracking_image,
                     frames.at(image.first).measurement.image,
                     cv::COLOR_BGR2GRAY);
      }
      frames.at(image.first).timeStamp = stampCorrected;
      frames.at(image.first).sensorId = image.first;
      frames.at(image.first).measurement.deliversKeypoints = false;
      useFrame = true;
    }
  }

  for(const auto & depthImage : depthImages) {
    if(parameters_.nCameraSystem.cameraType(depthImage.first).isUsed) {
      frames.at(depthImage.first).measurement.depthImage = depthImage.second;
      frames.at(depthImage.first).timeStamp = stampCorrected;
      frames.at(depthImage.first).sensorId = depthImage.first;
      frames.at(depthImage.first).measurement.deliversKeypoints = false;
    }
  }

  if(!useFrame) {
    return true; 
  }

  if (blocking_)
  {
    return cameraMeasurementsReceived_.PushBlockingIfFull(frames,1);
  }
  else
  {
    if(cameraMeasurementsReceived_.PushNonBlockingDroppingIfFull(frames, cameraInputQueueSize_)) {

      return false;
    }
    return true;
  }
}

bool ThreadedSlam::buildStereoPreview(
    const okvis::Time& stamp,
    const std::map<size_t, cv::Mat>& images,
    std::map<std::string, cv::Mat>& output_images)
{
  output_images.clear();
  const auto& tracking_system = parameters_.trackingCameraSystem();
  auto multi_frame = std::make_shared<okvis::MultiFrame>(
      tracking_system,
      stamp - Duration(parameters_.camera.image_delay),
      0);

  std::vector<size_t> camera_indices;
  camera_indices.reserve(images.size());
  for (const auto& image : images) {
    const size_t camera_index = image.first;
    if (camera_index >= tracking_system.numCameras()
        || !parameters_.nCameraSystem.isCameraConfigured(camera_index)
        || !parameters_.nCameraSystem.cameraType(camera_index).isUsed
        || image.second.empty()) {
      continue;
    }
    cv::Mat tracking_image = trackingImageDownsampler_.process(
        camera_index, image.second);
    if (tracking_image.channels() != 1) {
      cv::Mat gray;
      cv::cvtColor(tracking_image, gray, cv::COLOR_BGR2GRAY);
      tracking_image = std::move(gray);
    }
    multi_frame->setImage(camera_index, tracking_image);
    camera_indices.push_back(camera_index);
  }
  if (camera_indices.size() < 2) return false;

  const auto detect = [&](const size_t camera_index) {
    const auto T_SC = parameters_.nCameraSystem.T_SC(camera_index);
    if (!T_SC) return false;
    return frontend_.detectAndDescribe(
        camera_index, multi_frame, *T_SC, nullptr);
  };
  if (parameters_.frontend.parallelise_detection) {
    std::vector<std::thread> detection_threads;
    detection_threads.reserve(camera_indices.size() - 1);
    for (size_t i = 1; i < camera_indices.size(); ++i) {
      detection_threads.emplace_back(detect, camera_indices[i]);
    }
    const bool first_ok = detect(camera_indices.front());
    for (auto& thread : detection_threads) thread.join();
    if (!first_ok) return false;
  } else {
    for (const size_t camera_index : camera_indices) {
      if (!detect(camera_index)) return false;
    }
  }

  size_t left_camera = camera_indices.front();
  size_t right_camera = camera_indices[1];
  bool overlap_found = false;
  for (size_t i = 0; i < camera_indices.size() && !overlap_found; ++i) {
    for (size_t j = i + 1; j < camera_indices.size(); ++j) {
      if (multi_frame->hasOverlap(camera_indices[i], camera_indices[j])) {
        left_camera = camera_indices[i];
        right_camera = camera_indices[j];
        overlap_found = true;
        break;
      }
    }
  }

  std::map<size_t, std::vector<uint64_t>> match_ids;
  for (const size_t camera_index : camera_indices) {
    match_ids[camera_index].assign(
        multi_frame->numKeypoints(camera_index), uint64_t{0});
  }
  size_t match_count = 0;
  const size_t left_count = multi_frame->numKeypoints(left_camera);
  const size_t right_count = multi_frame->numKeypoints(right_camera);
  if (overlap_found && left_count > 0 && right_count > 0) {
    constexpr int kBriskDescriptorBytes = 48;
    cv::Mat left_descriptors(
        static_cast<int>(left_count), kBriskDescriptorBytes, CV_8UC1,
        const_cast<uchar*>(multi_frame->keypointDescriptor(left_camera, 0)));
    cv::Mat right_descriptors(
        static_cast<int>(right_count), kBriskDescriptorBytes, CV_8UC1,
        const_cast<uchar*>(multi_frame->keypointDescriptor(right_camera, 0)));
    std::vector<cv::DMatch> descriptor_matches;
    cv::BFMatcher(cv::NORM_HAMMING, false).match(
        left_descriptors, right_descriptors, descriptor_matches);
    const float threshold = static_cast<float>(
        frontend_.getBriskMatchingThreshold());
    for (const auto& match : descriptor_matches) {
      if (match.distance >= threshold) continue;
      const uint64_t match_id = static_cast<uint64_t>(++match_count);
      match_ids[left_camera][static_cast<size_t>(match.queryIdx)] = match_id;
      match_ids[right_camera][static_cast<size_t>(match.trainIdx)] = match_id;
    }
  }

  auto visualization = std::make_shared<ViVisualizer::VisualizationData>();
  visualization->currentFrames = multi_frame;
  visualization->T_WS = kinematics::Transformation::Identity();
  visualization->previewOnly = true;
  visualization->stereoDescriptorMatches = match_count;
  visualization->T_SCi.reserve(tracking_system.numCameras());
  for (size_t camera_index = 0;
       camera_index < tracking_system.numCameras(); ++camera_index) {
    const auto T_SC = tracking_system.T_SC(camera_index);
    visualization->T_SCi.push_back(
        T_SC ? *T_SC : kinematics::Transformation::Identity());
    for (size_t keypoint_index = 0;
         keypoint_index < multi_frame->numKeypoints(camera_index);
         ++keypoint_index) {
      Eigen::Vector2d measurement;
      double size = 0.0;
      multi_frame->getKeypoint(camera_index, keypoint_index, measurement);
      multi_frame->getKeypointSize(camera_index, keypoint_index, size);
      visualization->observations.emplace_back(
          keypoint_index,
          measurement,
          size,
          camera_index,
          0,
          Eigen::Vector4d::Zero(),
          match_ids[camera_index][keypoint_index],
          false);
    }
  }

  for (const size_t camera_index : camera_indices) {
    output_images[cameraDisplayName(parameters_, camera_index)] =
        visualizer_.drawMatches(visualization, camera_index);
  }
  return !output_images.empty();
}

bool ThreadedSlam::addImuMeasurement(const okvis::Time& stamp,
                                     const Eigen::Vector3d& alpha,
                                     const Eigen::Vector3d& omega)
{
  static bool warnOnce = true;
  if (!parameters_.imu.use) {
    if (warnOnce) {
      LOG(WARNING) << "imu measurement added, but IMU disabled";
      warnOnce = false;
    }
    return false;
  }
  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers.x() = parameters_.imu.s_a.x() * alpha.x();
  imu_measurement.measurement.accelerometers.y() = parameters_.imu.s_a.y() * alpha.y();
  imu_measurement.measurement.accelerometers.z() = parameters_.imu.s_a.z() * alpha.z();
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  const int imuQueueSize = 5000;

  if (blocking_)
  {
    return imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, size_t(imuQueueSize));
  }
  else
  {
    if(imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
         imu_measurement, size_t(imuQueueSize))) {
      LOG(WARNING) << "imu measurement drop ";
      return false;
    }
    return true;
  }

}

bool ThreadedSlam::addLidarMeasurement(const okvis::Time &stamp,
                                       const Eigen::Vector3d &rayMeasurement)
{
  if(!useAlignmentFactors_){
    return false;
  }

  okvis::LidarMeasurement lidarMeasurement;
  lidarMeasurement.measurement.rayMeasurement = rayMeasurement;
  lidarMeasurement.measurement.intensity = 0;
  lidarMeasurement.timeStamp = stamp;

  const int lidarQueueSize = 500000;
  sensorMeasurementDownsamplingCounter_ ++;
  bool drop = false;
  if(sensorMeasurementDownsamplingCounter_ % submapConfig_.sensorMeasurementDownsampling == 0){
    if (blocking_)
    {
      return lidarMeasurementsReceived_.PushBlockingIfFull(lidarMeasurement, size_t(lidarQueueSize));
    }
    else
    {
      drop = lidarMeasurementsReceived_.PushNonBlockingDroppingIfFull(
              lidarMeasurement, size_t(lidarQueueSize));
      if(drop)
        LOG(WARNING) << "lidar measurement drop ";
    }
    return !drop;
  }
  return true;
}

bool ThreadedSlam::addDepthMeasurement(const okvis::Time &stamp,
                                       const cv::Mat &depthImage) {
  CameraMeasurement depthMeasurement;
  depthMeasurement.timeStamp = stamp;
  depthMeasurement.measurement.depthImage = depthImage;
  return enqueueDepthMeasurement(std::move(depthMeasurement));
}

bool ThreadedSlam::addTaggedDepthMeasurement(
    const CameraMeasurement& measurement) {
  return enqueueDepthMeasurement(measurement);
}

bool ThreadedSlam::enqueueDepthMeasurement(
    CameraMeasurement depthMeasurement) {

  depthMeasurement.timeStamp =
      depthMeasurement.timeStamp - Duration(parameters_.camera.image_delay);

  if (parameters_.mappingInputPreprocessor
      && parameters_.mappingInputPreprocessor->enabled()) {
    OKVIS_ASSERT_TRUE(
        Exception,
        parameters_.mappingInputPreprocessor->process(
            parameters_.mappingInputPreprocessor->targetCameraIndex(),
            depthMeasurement,
            static_cast<size_t>(submapConfig_.depthImageResDownsampling)),
        "mapping input preprocessor rejected its configured target camera")
  }

  cv::Mat& depth_image = depthMeasurement.measurement.depthImage;
  if (!depth_image.empty()
      && (depth_image.type() != CV_32FC1
          || std::abs(submapConfig_.depthScalingFactor - 1.0f) > 1.0e-12f)) {

    cv::Mat converted;
    depth_image.convertTo(converted,
                          CV_32FC1,
                          submapConfig_.depthScalingFactor);
    depth_image = std::move(converted);
    depthMeasurement.measurement.depthBufferLifetime.reset();
  }

  sensorMeasurementDownsamplingCounter_ ++;
  bool drop = false;
  if(sensorMeasurementDownsamplingCounter_ % submapConfig_.sensorMeasurementDownsampling == 0){
    sensorMeasurementDownsamplingCounter_ = 0;
    if (blocking_) {
      return depthMeasurementsReceived_.PushBlockingIfFull(depthMeasurement, depthMeasurementQueueSize_);
    }
    else {
      drop = depthMeasurementsReceived_.PushNonBlockingDroppingIfFull(
          depthMeasurement, depthMeasurementQueueSize_);
      if (drop) {

      }
    }
    return !drop;
  }
  return true;
}

bool ThreadedSlam::addGpsMeasurement(const okvis::Time& stamp,
                                     const Eigen::Vector3d& pos,
                                     const Eigen::Vector3d& err)
{
  okvis::GpsMeasurement  gps_measurement;
  okvis::GpsSensorReadings gpsReading(pos, err(0), err(1), err(2));
  gps_measurement.timeStamp = stamp;
  gps_measurement.measurement = gpsReading;
  const int gpsQueueSize = 5000;

  if (blocking_)
  {
    return gpsMeasurementsReceived_.PushBlockingIfFull(gps_measurement, size_t(gpsQueueSize));
  }
  else
  {
    if(gpsMeasurementsReceived_.PushNonBlockingDroppingIfFull(gps_measurement, size_t(gpsQueueSize))) {
      LOG(WARNING) << "gps measurement drop ";
      return false;
    }

    return true;
  }

}

bool ThreadedSlam::addGeodeticGpsMeasurement(const okvis::Time& stamp,
                                             double lat, double lon, double height,
                                             double hAcc, double vAcc)
{

  okvis::GpsMeasurement  gps_measurement;
  okvis::GpsSensorReadings gpsReading(lat, lon, height, hAcc, vAcc);
  gps_measurement.timeStamp = stamp;
  gps_measurement.measurement = gpsReading;
  const int gpsQueueSize = 5000;

  if (blocking_)
  {
    return gpsMeasurementsReceived_.PushBlockingIfFull(gps_measurement, size_t(gpsQueueSize));
  }
  else
  {
    if(gpsMeasurementsReceived_.PushNonBlockingDroppingIfFull(gps_measurement, size_t(gpsQueueSize))) {
      LOG(WARNING) << "gps measurement drop ";
      return false;
    }

    return true;
  }

}

bool ThreadedSlam::addSubmapAlignmentConstraints(const SupereightMapType* submap_A_ptr,
                                                 const SupereightMapType* submap_B_ptr,
                                                 const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                                                 std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                                 std::vector<float> sensorError) {

  submapAlignmentFactorsReceived_.PushNonBlocking(
          AlignmentTerm(submap_A_ptr, submap_B_ptr,frame_A_id,frame_B_id, pointCloud, sensorError));

  return true;
}

void ThreadedSlam::setBlocking(bool blocking)
{
  blocking_ = blocking;

  if(blocking_)
  {

  }
}

void ThreadedSlam::enqueueGsOptimizedPoses(
    const std::map<uint64_t, Eigen::Matrix4f>& poses_by_state_id)
{
  if (poses_by_state_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(pendingGsPoseUpdatesMtx_);
  for (const auto& pose_kv : poses_by_state_id) {
    pendingGsPoseUpdates_[pose_kv.first] = pose_kv.second;
  }
}

void ThreadedSlam::applyPendingGsPoseUpdates()
{
  AlignedMap<uint64_t, Eigen::Matrix4f> pending_updates;
  {
    std::lock_guard<std::mutex> lock(pendingGsPoseUpdatesMtx_);
    if (pendingGsPoseUpdates_.empty()) {
      return;
    }
    pending_updates.swap(pendingGsPoseUpdates_);
  }

  size_t applied = 0;
  for (const auto& pose_kv : pending_updates) {
    const StateId state_id(pose_kv.first);
    if (!state_id.isInitialised()) {
      continue;
    }
    const Eigen::Matrix4d T_WS_d = pose_kv.second.cast<double>();
    if (estimator_.setPose(state_id, kinematics::TransformationCacheless(T_WS_d))) {
      affectedStates_.insert(state_id);
      applied++;
    }
  }
  (void) applied;
}

bool ThreadedSlam::getNextFrame(MultiFramePtr &multiFrame) {
  std::vector<okvis::CameraMeasurement> frames;
  if(!cameraMeasurementsReceived_.getCopyOfFront(&frames)) {
    return false;
  }
  const size_t numCameras = frames.size();
  multiFrame.reset(new okvis::MultiFrame(
      parameters_.trackingCameraSystem(), frames.at(0).timeStamp, 0));
  bool success = false;
  for(size_t im = 0; im < numCameras; ++im)
  {
    cv::Mat filtered = frames.at(im).measurement.image;
    if(!filtered.empty()) {
      success = true; 
    }
    multiFrame->setImage(im, filtered);
    multiFrame->setDepthImage(im, frames.at(im).measurement.depthImage);
  }
  return success;
}

bool ThreadedSlam::processFrame() {
  MultiFramePtr multiFrame;
  ImuMeasurement imuMeasurement;
  LidarMeasurement lidarMeasurement;
  GpsMeasurement gpsMeasurement;
  CameraMeasurement depthMeasurement;
  const size_t numCameras = parameters_.nCameraSystem.numCameras();
  double trackingLatencySec = 0.0;

  kinematics::Transformation T_WS;
  SpeedAndBias speedAndBias;

  bool ranDetection = false;
  if(firstFrame_) {

    if(parameters_.imu.use && !imuMeasurementsReceived_.getCopyOfFront(&imuMeasurement)) {
      return false;
    }

    if(!getNextFrame(multiFrame)) {
      return false;
    }

    if(parameters_.imu.use) {
      if(multiFrame->timestamp()-Duration(imuTemporalOverlap) <= imuMeasurement.timeStamp) {

        LOG(WARNING) << "startup: dropping frame because IMU measurements are newer, t="
                     << imuMeasurement.timeStamp;
        std::vector<okvis::CameraMeasurement> frames;
        cameraMeasurementsReceived_.PopBlocking(&frames);
        return false;
      }

      if(!imuMeasurementsReceived_.getCopyOfBack(&imuMeasurement)) {
        return false;
      }
      if(imuMeasurement.timeStamp < multiFrame->timestamp() + Duration(imuTemporalOverlap)) {
        return false; 
      }

      do {
        if(imuMeasurementsReceived_.PopNonBlocking(&imuMeasurement))
        {
          imuMeasurementDeque_.push_back(imuMeasurement);
        } else {
          return false;
        }
      } while(imuMeasurementDeque_.back().timeStamp
              < multiFrame->timestamp() + Duration(imuTemporalOverlap));
    }

    while(!lidarMeasurementsReceived_.Empty()
      && lidarMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp()){
      lidarMeasurementsReceived_.PopBlocking(&lidarMeasurement);
    }

    while(!gpsMeasurementsReceived_.Empty() && gpsMeasurementsReceived_.queue_.front().timeStamp < multiFrame ->timestamp()){
        gpsMeasurementsReceived_.PopBlocking(&gpsMeasurement);
    } 

    firstFrame_ = false;
  } else {

    if(!getNextFrame(multiFrame)) {
      if(optimisationThread_.joinable()) {

        optimisationThread_.join();
      }

      applyPendingGsPoseUpdates();

      return false;
    }

    if(parameters_.imu.use) {
      while(!shutdown_ && imuMeasurementDeque_.back().timeStamp <
            multiFrame->timestamp() + Duration(imuTemporalOverlap))
      {
        if(imuMeasurementsReceived_.PopNonBlocking(&imuMeasurement))
        {
          imuMeasurementDeque_.push_back(imuMeasurement);
        } else {
          return false;
        }
      }
    }
  }
  if (!lastOptimisedState_.id.isInitialised()) {

    if (parameters_.imu.use) {
      bool success = ceres::ImuError::initPose(imuMeasurementDeque_, T_WS);
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            success,
                            "pose could not be initialized from imu measurements.")
      (void) (success); 
    } else {

      kinematics::Transformation T_WC;
      T_WC.set(T_WC.r(), T_WC.q() * Eigen::Quaterniond(-sqrt(2), sqrt(2), 0, 0));
      T_WS = T_WC * parameters_.nCameraSystem.T_SC(0)->inverse();
    }

    const auto detectStageStart = std::chrono::steady_clock::now();
    TimerSwitchable detectTimer("1 DetectAndDescribe");
    if(parameters_.frontend.parallelise_detection) {
      std::vector<std::shared_ptr<std::thread>> detectionThreads;
      for(size_t im = 1; im < numCameras; ++im) {
        kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
        if(!multiFrame->image(im).empty()) {
          detectionThreads.emplace_back(
              new std::thread(&Frontend::detectAndDescribe, &frontend_,
                              im,  multiFrame, T_WC, nullptr));
        }
      }
      kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
      if(!multiFrame->image(0).empty()) {
        frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr);
      }
      for(size_t i = 0; i < detectionThreads.size(); ++i) {
        detectionThreads[i]->join();
      }
    } else {
      for(size_t im = 0; im < numCameras; ++im) {
        kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
        if(!multiFrame->image(im).empty()) {
          frontend_.detectAndDescribe(im,  multiFrame, T_WC, nullptr);
        }
      }
    }
    if(!frontend_.isInitialized() && multiFrame->numKeypoints() < 15) {
      LOG(WARNING) << "Not enough keypoints (" << multiFrame->numKeypoints()
                   << ") -- cannot initialise yet.";
      std::vector<okvis::CameraMeasurement> frames;
      cameraMeasurementsReceived_.PopBlocking(&frames);
      return false;
    }
    ranDetection = true;
    detectTimer.stop();
    trackingLatencySec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - detectStageStart)
            .count();

  } else {

    if (parameters_.imu.use) {

      T_WS = lastOptimisedState_.T_WS;
      kinematics::Transformation T_WS_prev = T_WS;
      speedAndBias.head<3>() = lastOptimisedState_.v_W;
      speedAndBias.segment<3>(3) = lastOptimisedState_.b_g;
      speedAndBias.tail<3>() = lastOptimisedState_.b_a;
      ceres::ImuError::propagation(imuMeasurementDeque_,
                                           parameters_.imu,
                                           T_WS,
                                           speedAndBias,
                                           lastOptimisedState_.timestamp,
                                           multiFrame->timestamp());
    } else {
      T_WS = lastOptimisedState_.T_WS;
      if (preLastOptimisedState_.id.isInitialised()) {
        const double r = (multiFrame->timestamp() - lastOptimisedState_.timestamp).toSec()
        / (lastOptimisedState_.timestamp - preLastOptimisedState_.timestamp).toSec();
        kinematics::Transformation T_WS_m1 = preLastOptimisedState_.T_WS;
        Eigen::Vector3d dr = r * (T_WS.r() - T_WS_m1.r());
        Eigen::AngleAxisd daa(T_WS.q() * T_WS_m1.q().inverse());
        daa.angle() *= r;
        T_WS.set(T_WS.r() + dr, T_WS.q() * Eigen::Quaterniond(daa));
      }
    }

    while(!shutdown_ && !gpsMeasurementsReceived_.Empty() && gpsMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp())
    {
        if(gpsMeasurementsReceived_.PopBlocking(&gpsMeasurement))
        {
          gpsMeasurementDeque_.push_back(gpsMeasurement);
        }
    }

    while(!shutdown_ && !lidarMeasurementsReceived_.Empty() && lidarMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp())
    {
      if(lidarMeasurementsReceived_.PopBlocking(&lidarMeasurement))
      {
        lidarMeasurementDeque_.push_back(lidarMeasurement);
      }
    }

    while(!shutdown_ && !depthMeasurementsReceived_.Empty() && depthMeasurementsReceived_.queue_.front().timeStamp <= multiFrame->timestamp())
    {
      if (depthMeasurementsReceived_.PopBlocking(&depthMeasurement))
      {
        depthMeasurementDeque_.push_back(depthMeasurement);
      }
    }
  }

  std::vector<okvis::CameraMeasurement> frames;
  cameraMeasurementsReceived_.PopBlocking(&frames);

  if(!ranDetection) {
    const auto detectStageStart = std::chrono::steady_clock::now();
    TimerSwitchable detectTimer("1 DetectAndDescribe");
    if(parameters_.frontend.parallelise_detection) {
      std::vector<std::shared_ptr<std::thread>> detectionThreads;
      for(size_t im = 1; im < numCameras; ++im) {
        kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
        if(!multiFrame->image(im).empty()) {
          detectionThreads.emplace_back(
              new std::thread(&Frontend::detectAndDescribe, &frontend_,
                              im,  multiFrame, T_WC, nullptr));
        }
      }
      kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
      if(!multiFrame->image(0).empty()) {
        frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr);
      }
      for(size_t i = 0; i < detectionThreads.size(); ++i) {
        detectionThreads[i]->join();
      }
    } else {
      for(size_t im = 0; im < numCameras; ++im) {
        kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
        if(!multiFrame->image(im).empty()) {
          frontend_.detectAndDescribe(im,  multiFrame, T_WC, nullptr);
        }
      }
    }
    if(!frontend_.isInitialized() && multiFrame->numKeypoints() < 15) {
      LOG(WARNING) << "Not enough keypoints (" << multiFrame->numKeypoints()
                   << ") -- cannot initialise yet.";
      return true;
    }
    detectTimer.stop();
    trackingLatencySec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - detectStageStart)
            .count();
  }

  if(optimisationThread_.joinable()) {

    optimisationThread_.join();
  }

  applyPendingGsPoseUpdates();

  if (estimator_.numFrames() > 0) {
    const StateId currentId = estimator_.currentStateId();
    lastOptimisedState_.T_WS = estimator_.pose(currentId);
    SpeedAndBias speedAndBias = estimator_.speedAndBias(currentId);
    lastOptimisedState_.v_W = speedAndBias.head<3>();
    lastOptimisedState_.b_g = speedAndBias.segment<3>(3);
    lastOptimisedState_.b_a = speedAndBias.tail<3>();
    lastOptimisedState_.id = currentId;
    lastOptimisedState_.timestamp = estimator_.timestamp(currentId);
    if (estimator_.numFrames() > 1) {
      const StateId previousId(currentId.value() - 1);
      preLastOptimisedState_.T_WS = estimator_.pose(previousId);
      preLastOptimisedState_.id = previousId;
      preLastOptimisedState_.timestamp = estimator_.timestamp(previousId);
    }
  }

  if(shutdown_) {
    return false;
  }

  if (estimator_.numFrames() > 0) {
    if(parameters_.imu.use) {
      while (!shutdown_
             && (imuMeasurementDeque_.front().timeStamp
                 < lastOptimisedState_.timestamp - Duration(imuTemporalOverlap))) {
        auto imuMeasurement = imuMeasurementDeque_.front();
        imuMeasurementDeque_.pop_front();
        if (imuMeasurementDeque_.empty()
            || (imuMeasurementDeque_.front().timeStamp
                > lastOptimisedState_.timestamp - Duration(imuTemporalOverlap))) {
          imuMeasurementDeque_.push_front(imuMeasurement); 
          break; 
        }
      }
    }
  }

  Time matchingStart = Time::now();
  TimerSwitchable matchTimer("2 Match");
  bool asKeyframe = false;

  if (!estimator_.addStates(multiFrame, imuMeasurementDeque_, asKeyframe)) {
    LOG(ERROR)<< "Failed to add state! will drop multiframe.";
    matchTimer.stop();
    return true;
  }
  else{

    if(useAlignmentFactors_)
    {
      TimerSwitchable tProcessLiveDepthLidar("8 Processing data for map-to-frame factors");

      kinematics::Transformation T_WS_live = estimator_.pose(StateId(multiFrame->id()));
      if(parameters_.lidar) {
        if(previousSubmap_){
          TimerSwitchable tLiveUndistortion("8.1 Live undistortion");
          LidarMotionUndistortion motionUndistortion(lastOptimisedState_, T_WS_live, T_SD_,
                                                    lidarMeasurementDeque_, imuMeasurementDeque_);
          motionUndistortion.deskew();
          tLiveUndistortion.stop();

          TimerSwitchable tFilterObserved("8.2 Filter observed points");
          size_t observed_points = motionUndistortion.filterObserved(previousSubmap_, estimator_.pose(StateId(previousSubmapId_)));
          if(observed_points < submapConfig_.numSubmapFactors){
              noOverlapCounter_++;
          }
          else{
              noOverlapCounter_ = 0;
          }
          tFilterObserved.stop();
          TimerSwitchable tDownsampling("8.3 Downsampling");
          motionUndistortion.downsample(submapConfig_.numSubmapFactors, submapConfig_.voxelGridResolution);
          tDownsampling.stop();

          std::vector<float> sensorErrors(motionUndistortion.deskewedDownsampledPointCloud().size());
          std::fill(sensorErrors.begin(), sensorErrors.end(), submapConfig_.sensorError);
          estimator_.addSubmapAlignmentConstraints(
                  previousSubmap_, previousSubmapId_, multiFrame->id(),
                  motionUndistortion.deskewedDownsampledPointCloud(), sensorErrors, true, "Tukey");
        }
      }
      else {
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> livePoints;
        std::vector<float> liveSigmas;
        computeLiveDepthMeasurements(T_WS_live, multiFrame->timestamp(), livePoints, liveSigmas);

        if (livePoints.size() > 0.05*submapConfig_.numSubmapFactors) {

          estimator_.addSubmapAlignmentConstraints(previousSubmap_, previousSubmapId_, multiFrame->id(),
                                                    livePoints, liveSigmas, false, "Tukey");
          float sigmaMean = 0;
          for (size_t ii = 0; ii < liveSigmas.size(); ii++) {
            sigmaMean += liveSigmas[ii];
          }
          DLOG(INFO) << "Map-to-frame factor added "<< previousSubmapId_ << "-" 
            << multiFrame->id() << " with " << livePoints.size()
            << " with uncertainty " << sigmaMean / liveSigmas.size() << std::endl;

        }  
      }
      tProcessLiveDepthLidar.stop();
    }
  }
  imuMeasurementsByFrame_[StateId(multiFrame->id())] = imuMeasurementDeque_;

  bool kfPrior = false;
  if(submapConfig_.useMap2LiveFactors && parameters_.lidar){
    kfPrior = needsNewLidarKeyframe();
    if(kfPrior) {
      lidarKeyframes_.insert(StateId(multiFrame->id()));
    }
  }

  const auto solveStageStart = std::chrono::steady_clock::now();
  if(!frontend_.dataAssociationAndInitialization(
        estimator_, parameters_, multiFrame, kfPrior, &asKeyframe) && !frontend_.isInitialized()) {
    LOG(WARNING) << "Not enough matches, cannot initialise";
    frontend_.clear();
    estimator_.clear();
    return false;
  }
  estimator_.setKeyframe(StateId(multiFrame->id()), asKeyframe);
  matchTimer.stop();
  trackingLatencySec +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - solveStageStart)
          .count();

  if (trackingLatencySec > 1.0e-6) {
    const float instFps = static_cast<float>(1.0 / trackingLatencySec);
    const float prevFps = frontendTrackingFpsHz_.load(std::memory_order_relaxed);
    constexpr float kAlpha = 0.25f;
    const float filteredFps =
        (prevFps > 0.0f) ? ((1.0f - kAlpha) * prevFps + kAlpha * instFps) : instFps;
    frontendTrackingFpsHz_.store(filteredFps, std::memory_order_relaxed);
  }

  estimator_.addGpsMeasurementsOnAllGraphs(gpsMeasurementDeque_, imuMeasurementDeque_);

  while(!shutdown_ && !gpsMeasurementDeque_.empty() && gpsMeasurementDeque_.front().timeStamp < multiFrame->timestamp() )
  {
    gpsMeasurementDeque_.pop_front();
  }

  while(!shutdown_ && !lidarMeasurementDeque_.empty() && lidarMeasurementDeque_.front().timeStamp < multiFrame->timestamp() )
  {
    lidarMeasurementDeque_.pop_front();
  }

  while(!shutdown_ && !depthMeasurementDeque_.empty() && depthMeasurementDeque_.front().timeStamp <= multiFrame->timestamp() ) {
    depthMeasurementDeque_.pop_front();
  }

  AlignmentTerm alignmentTerm;
  while(!shutdown_ && !submapAlignmentFactorsReceived_.Empty())
  {
    submapAlignmentFactorsReceived_.PopNonBlocking(&alignmentTerm);

    if(parameters_.lidar) {

      estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
                                               alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
                                               alignmentTerm.sensorError, true, "Tukey");

      previousSubmap_ = alignmentTerm.submap_B_ptr;
      previousSubmapId_ = alignmentTerm.frame_B_id;
    }
    else {

      if (alignmentTerm.submap_A_ptr) {
        estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
                                                alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
                                                alignmentTerm.sensorError, false, "Tukey");

      }
      if (alignmentTerm.frame_A_id != 0) {
        previousSubmap_ = alignmentTerm.submap_A_ptr;
        previousSubmapId_ = alignmentTerm.frame_A_id;
      }
      else {
        previousSubmap_ = alignmentTerm.submap_B_ptr;
        previousSubmapId_ = alignmentTerm.frame_B_id;
      }
    }
    LOG(INFO) << "[ThreadedSlam] Registered new reference submap with ID " << previousSubmapId_;
  }

  Eigen::Vector3d gyr(0,0,0);
  auto riter = imuMeasurementDeque_.rbegin();
  while(riter!=imuMeasurementDeque_.rend() && riter->timeStamp > multiFrame->timestamp()) {
    gyr = riter->measurement.gyroscopes;
    ++riter;
  }
  Time now = Time::now();
  double dt = parameters_.estimator.realtime_time_limit-(now-matchingStart).toSec();
  if(dt < 0.0) {
    dt = 0.01;
  }
  if(parameters_.estimator.enforce_realtime) {
    estimator_.setOptimisationTimeLimit(dt,
          parameters_.estimator.realtime_min_iterations);
  }
  optimisationThread_ = std::thread(&ThreadedSlam::optimisePublishMarginalise,
                                    this, multiFrame, gyr);

  if(estimator_.needsFullGraphOptimisation()
     ) {
    if(fullGraphOptimisationThread_.joinable()) {
      fullGraphOptimisationThread_.join();
    }

    fullGraphOptimisationThread_ = std::thread(
          &ViSlamBackend::optimiseFullGraph, &estimator_,
          parameters_.estimator.full_graph_iterations,
          std::ref(posegraphOptimisationSummary_),
          parameters_.estimator.full_graph_num_threads, false,
          ViSlamBackend::FinalBaProgressCallback{}, 0);
  }

  return true;
}

void ThreadedSlam::optimisePublishMarginalise(MultiFramePtr multiFrame,
                                              const Eigen::Vector3d& gyroReading) {
  kinematics::Transformation T_WS;
  SpeedAndBias speedAndBiases;
  const size_t numCameras = parameters_.nCameraSystem.numCameras();

  TimerSwitchable optimiseTimer("3 Optimise");
  std::vector<StateId> updatedStatesRealtime;
  estimator_.optimiseRealtimeGraph(
      parameters_.estimator.realtime_max_iterations, updatedStatesRealtime,
      parameters_.estimator.realtime_num_threads,
      false, false, frontend_.isInitialized());
  optimiseTimer.stop();

  std::vector<StateId> updatedStatesSync;
  if(estimator_.isLoopClosureAvailable()) {
    OKVIS_ASSERT_TRUE(Exception,
                      !estimator_.isLoopClosing(),
                      "loop closure available, but still loop closing -- bug")
    TimerSwitchable synchronisationTimer("5 Import full optimisation");
    estimator_.synchroniseRealtimeAndFullGraph(updatedStatesSync);
    synchronisationTimer.stop();
  }

  TimerSwitchable publishTimer("4 Prepare publishing");
  T_WS = estimator_.pose(StateId(multiFrame->id()));
  speedAndBiases = estimator_.speedAndBias(StateId(multiFrame->id()));
  StateId id(multiFrame->id());
  State state;
  state.id = id;
  state.T_WS = T_WS;
  state.v_W = speedAndBiases.head<3>();
  state.b_g = speedAndBiases.segment<3>(3);
  state.b_a = speedAndBiases.tail<3>();
  state.omega_S = gyroReading - speedAndBiases.segment<3>(3);
  state.timestamp = multiFrame->timestamp();
  state.previousImuMeasurements = imuMeasurementsByFrame_.at(id);
  state.isKeyframe = estimator_.isKeyframe(id);
  state.isOnlineExtrinsics = parameters_.camera.online_calibration.do_extrinsics;
  if (state.isOnlineExtrinsics && depthCamera_) {
    for (size_t camIndex = 0; camIndex < numCameras; ++camIndex) {
      if (parameters_.nCameraSystem.cameraType(camIndex).isUsedMapping
          && (!parameters_.mappingInputPreprocessor
              || !parameters_.mappingInputPreprocessor->enabled()
              || parameters_.mappingInputPreprocessor->targetCameraIndex()
                     == camIndex)) {
        state.extrinsics[camIndex] = estimator_.extrinsics(id, camIndex);
        T_SD_ = state.extrinsics[camIndex]; 
      }
    }
  }

  AlignedMap<uint64_t, kinematics::Transformation> T_AiW;
  if (estimator_.T_AiS_.count(id)) {
    AlignedMap<uint64_t, kinematics::Transformation> T_AiS = estimator_.T_AiS_.at(id);
    for (const auto &T_AS : T_AiS) {
      T_AiW[T_AS.first] = T_AS.second * T_WS.inverse();
    }
  }
  state.T_AiW = T_AiW;
  estimator_.getObservedIds(state.id, state.covisibleFrameIds);

  TrackingState trackingState;
  if(lidarKeyframes_.count(id) == 0){
    trackingState.isLidarKeyframe = false;
  }
  else{
    trackingState.isLidarKeyframe = true;
  }
  trackingState.id = id;
  trackingState.isKeyframe = estimator_.isKeyframe(id);
  trackingState.recognisedPlace = estimator_.closedLoop(id);
  const double trackingQuality = estimator_.trackingQuality(id);
  if(trackingQuality < 0.01) {
    trackingState.trackingQuality = TrackingQuality::Lost;
  } else if (trackingQuality < 0.3){
    trackingState.trackingQuality = TrackingQuality::Marginal;
  } else {
    trackingState.trackingQuality = TrackingQuality::Good;
  }
  trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(id, false);

  hasStarted_.store(true);

  if(optimisedGraphCallback_) {

    std::vector<StateId> updatedStateIds;
    PublicationData publicationData;
    publicationData.state = state;
    publicationData.trackingState = trackingState;
    publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
    if(updatedStatesSync.size()>0) {
      updatedStateIds = updatedStatesSync;
    } else {
      updatedStateIds = updatedStatesRealtime;
    }
    for(const auto & id : updatedStateIds) {
      kinematics::Transformation T_WS = estimator_.pose(id);
      SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
      Time timestamp = estimator_.timestamp(id);
      const bool isKeyframe = estimator_.isKeyframe(id);
      ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
      Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
      for(auto riter = imuMeasurements.rbegin(); riter!=imuMeasurements.rend(); ++riter) {
        if(riter->timeStamp < timestamp) {
          omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
          break;
        }
      }

      std::set<StateId> observedIds;
      estimator_.getObservedIds(id, observedIds);
      AlignedVector<Eigen::Vector3d> gpsPoints;
      estimator_.gpsMeasurements(id, gpsPoints);
      (*publicationData.updatedStates)[id] = State{T_WS, speedAndBias.head<3>(),
                                    speedAndBias.segment<3>(3), speedAndBias.tail<3>(),
                                    omega_S, timestamp, id, imuMeasurements, isKeyframe,
                                    AlignedMap<uint64_t, kinematics::Transformation>(),
                                    observedIds, true,
                                    false, AlignedMap<uint64_t, kinematics::Transformation>(),
                                    estimator_.T_GW(), gpsPoints};
    }
    for (const auto &id : affectedStates_) {
      kinematics::Transformation T_WS = estimator_.pose(id);
      SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
      Time timestamp = estimator_.timestamp(id);
      ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
      Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
      for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
        if (riter->timeStamp < timestamp) {
          omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
          break;
        }
      }
      std::set<StateId> observedIds;
      estimator_.getObservedIds(id, observedIds);
      AlignedVector<Eigen::Vector3d> gpsPoints;
      estimator_.gpsMeasurements(id, gpsPoints);
      (*publicationData.updatedStates)[id]
        = State{T_WS,
                speedAndBias.head<3>(),
                speedAndBias.segment<3>(3),
                speedAndBias.tail<3>(),
                omega_S,
                timestamp,
                id,
                imuMeasurements,
                estimator_.isKeyframe(id),
                AlignedMap<uint64_t, kinematics::Transformation>(),
                observedIds, false,
                false, AlignedMap<uint64_t, kinematics::Transformation>(),
                estimator_.T_GW(), gpsPoints};
    }
    affectedStates_.clear();

    publicationData.landmarksPublish.reset(new MapPointVector());
    MapPoints landmarks;
    estimator_.getLandmarks(landmarks);
    publicationData.landmarksPublish->reserve(landmarks.size());
    for(const auto & lm : landmarks) {
      auto latestObservedFrameId = *lm.second.observations.rbegin();
      publicationData.landmarksPublish->push_back(
            MapPoint(lm.first.value(), lm.second.point, lm.second.quality,
                     latestObservedFrameId.getFrameId()));
    }


    if(blocking_){

      publicationQueue_.PushBlockingIfFull(publicationData,1);
    }
    else{
      const bool overrun = publicationQueue_.PushNonBlockingDroppingIfFull(publicationData,3);
      if(overrun) {
        LOG(ERROR) << "publication (full update) overrun: dropping";
      }
    }

    std::set<StateId> affectedStateIds;
    trajectory_.update(publicationData.trackingState, publicationData.updatedStates, affectedStateIds);
  }
  publishTimer.stop();

  TimerSwitchable visualisationTimer("6 Visualising");
  if(parameters_.output.display_overhead) {
    TimerSwitchable visualisation1Timer("6.1 Visualising overhead");

    cv::Mat image(580, 580, CV_8UC3);
    image.setTo(cv::Scalar(10, 10, 10));
    estimator_.drawOverheadImage(image);
    overheadImages_.PushNonBlockingDroppingIfFull(image,1);
    visualisation1Timer.stop();
  }
  if(parameters_.output.display_matches) {
    TimerSwitchable visualisation2Timer("6.2 Prepare visualising matches");

    ViVisualizer::VisualizationData::Ptr visualizationDataPtr(
          new ViVisualizer::VisualizationData());
    visualizationDataPtr->observations.resize(multiFrame->numKeypoints());
    okvis::MapPoint2 landmark;
    okvis::ObservationVector::iterator it = visualizationDataPtr
        ->observations.begin();
    for (size_t camIndex = 0; camIndex < numCameras; ++camIndex) {
      visualizationDataPtr->T_SCi.push_back(
        estimator_.extrinsics(estimator_.currentStateId(), camIndex));
      for (size_t k = 0; k < multiFrame->numKeypoints(camIndex); ++k) {
        OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),
                              "Observation-vector not big enough")
        it->keypointIdx = k;
        multiFrame->getKeypoint(camIndex, k, it->keypointMeasurement);
        multiFrame->getKeypointSize(camIndex, k, it->keypointSize);
        it->cameraIdx = camIndex;
        it->frameId = multiFrame->id();
        it->landmarkId = multiFrame->landmarkId(camIndex, k);
        if (estimator_.isLandmarkAdded(LandmarkId(it->landmarkId)) &&
            estimator_.isObserved(KeypointIdentifier(it->frameId, camIndex, k))) {
          estimator_.getLandmark(LandmarkId(it->landmarkId), landmark);
          it->landmark_W = landmark.point;
          if (estimator_.isLandmarkInitialised(LandmarkId(it->landmarkId)))
            it->isInitialized = true;
          else
            it->isInitialized = false;
        } else {
          it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);

        }
        ++it;
      }
    }
    visualizationDataPtr->T_WS = estimator_.pose(estimator_.currentStateId());
    visualizationDataPtr->currentFrames = multiFrame;
    visualizationDataPtr->isKeyframe = trackingState.isKeyframe;
    visualizationDataPtr->recognisedPlace = trackingState.recognisedPlace;
    if(trackingState.trackingQuality == TrackingQuality::Lost) {
      visualizationDataPtr->trackingQuality =
          ViVisualizer::VisualizationData::TrackingQuality::Lost;
    } else if(trackingState.trackingQuality == TrackingQuality::Marginal) {
      visualizationDataPtr->trackingQuality =
          ViVisualizer::VisualizationData::TrackingQuality::Marginal;
    }
    visualisation2Timer.stop();
    visualisationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr,1);
  }
  visualisationTimer.stop();

  bool expand = true;
  TimerSwitchable marginaliseTimer("7 Marginalise");
  estimator_.applyStrategy(
        size_t(parameters_.estimator.num_keyframes),
        size_t(parameters_.estimator.num_loop_closure_frames),
    size_t(parameters_.estimator.num_imu_frames), affectedStates_, expand);
  marginaliseTimer.stop();

}

bool ThreadedSlam::needsNewLidarKeyframe()
{
  if(previousSubmap_){
    if(noOverlapCounter_ > 1){
      DLOG(INFO) << "Manually triggering new KF for lidar tracking!!! For lastOptimisedState.id = " << lastOptimisedState_.id.value();
      noOverlapCounter_ = 0;
      return true;
    }
  }
  return false;
}

void ThreadedSlam::visualisationLoop()
{
  while (!shutdown_) {
    ViVisualizer::VisualizationData::Ptr visualisationData;
    if(visualisationData_.PopBlocking(&visualisationData)) {
      std::vector<cv::Mat> outImages(parameters_.nCameraSystem.numCameras());
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        if(parameters_.nCameraSystem.cameraType(i).isUsed
           && !visualisationData->currentFrames->image(i).empty()) {
          outImages[i] = visualizer_.drawMatches(visualisationData, i);
        }
      }
      visualisationImages_.PushNonBlockingDroppingIfFull(outImages,1);
    } else {
      return;
    }
  }
}

void ThreadedSlam::publishingLoop()
{
  while (!shutdown_) {
    PublicationData publicationData;
    if(publicationQueue_.PopBlocking(&publicationData)) {
      if(optimisedGraphCallback_) {
        optimisedGraphCallback_(publicationData.state, publicationData.trackingState,
                                publicationData.updatedStates, publicationData.landmarksPublish);
      }
    } else {
      return;
    }
  }
}

void ThreadedSlam::display(std::map<std::string, cv::Mat> &images)
{

  std::vector<cv::Mat> outImages(parameters_.nCameraSystem.numCameras());
  if(visualisationImages_.PopNonBlocking(&outImages)) {

    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); i++) {
      if(!outImages[i].empty()) {
        images[cameraDisplayName(parameters_, i)] = outImages[i];
      }
    }
  }

  cv::Mat topDebugImg;
  if(overheadImages_.PopNonBlocking(&topDebugImg)) {
    if(!topDebugImg.empty()) {
      images["Top Debug View"] = topDebugImg;
    }
  }
}

void ThreadedSlam::stopThreading() {
  if (shutdown_) {
    return;
  }

  while (processFrame()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  frontend_.endCnnThreads();

  if(optimisationThread_.joinable()) {
    optimisationThread_.join(); 
  }

  imuMeasurementsReceived_.Shutdown();
  cameraMeasurementsReceived_.Shutdown();
  gpsMeasurementsReceived_.Shutdown();
  lidarMeasurementsReceived_.Shutdown();
  submapAlignmentFactorsReceived_.Shutdown();

  visualisationImages_.Shutdown();
  visualisationData_.Shutdown();
  publicationQueue_.Shutdown();
  lastOptimisedQueue_.Shutdown();

  if(fullGraphOptimisationThread_.joinable()) {
    LOG(INFO) << "wait for background optimisation to finish...";
    fullGraphOptimisationThread_.join();

    if (estimator_.isLoopClosureAvailable()) {
      OKVIS_ASSERT_TRUE(Exception,
                        !estimator_.isLoopClosing(),
                        "loop closure available, but still loop closing -- bug")
      LOG(INFO) << "import final background optimisation and publish...";
      TimerSwitchable synchronisationTimer("5 Import full optimisation");
      std::vector<StateId> updatedStates;
      estimator_.synchroniseRealtimeAndFullGraph(updatedStates);
      synchronisationTimer.stop();

      StateId currentId = estimator_.currentStateId();
      kinematics::Transformation T_WS = estimator_.pose(currentId);
      SpeedAndBias speedAndBiases = estimator_.speedAndBias(currentId);
      State state;
      state.id = currentId;
      state.T_WS = T_WS;
      state.v_W = speedAndBiases.head<3>();
      state.b_g = speedAndBiases.segment<3>(3);
      state.b_a = speedAndBiases.tail<3>();
      state.omega_S.setZero(); 
      state.timestamp = estimator_.timestamp(currentId);
      state.previousImuMeasurements = imuMeasurementsByFrame_.at(currentId);
      state.isKeyframe = estimator_.isKeyframe(currentId);
      TrackingState trackingState;
      trackingState.id = currentId;
      trackingState.isKeyframe = estimator_.isKeyframe(currentId);
      trackingState.recognisedPlace = estimator_.closedLoop(currentId);
      const double trackingQuality = estimator_.trackingQuality(currentId);
      if (trackingQuality < 0.01) {
        trackingState.trackingQuality = TrackingQuality::Lost;
      } else if (trackingQuality < 0.3) {
        trackingState.trackingQuality = TrackingQuality::Marginal;
      } else {
        trackingState.trackingQuality = TrackingQuality::Good;
      }
      trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(currentId, false);
      hasStarted_.store(true);

      if (optimisedGraphCallback_) {

        PublicationData publicationData;
        publicationData.state = state;
        publicationData.trackingState = trackingState;
        publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
        for (const auto &id : updatedStates) {
          kinematics::Transformation T_WS = estimator_.pose(id);
          SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
          Time timestamp = estimator_.timestamp(id);
          ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
          Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
          for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
            if (riter->timeStamp < timestamp) {
              omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
              break;
            }
          }
          std::set<StateId> observedIds;
          estimator_.getObservedIds(id, observedIds);
          (*publicationData.updatedStates)[id] =
            State{T_WS, speedAndBias.head<3>(), speedAndBias.segment<3>(3), speedAndBias.tail<3>(),
                  omega_S, timestamp, id, imuMeasurements, estimator_.isKeyframe(id),
                  AlignedMap<uint64_t, kinematics::Transformation>(), observedIds, true,
                  false, AlignedMap<uint64_t, kinematics::Transformation>(),
                  kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
          affectedStates_.erase(id); 
        }
        for (const auto &id : affectedStates_) {
          kinematics::Transformation T_WS = estimator_.pose(id);
          SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
          Time timestamp = estimator_.timestamp(id);
          ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
          Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
          for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
            if (riter->timeStamp < timestamp) {
              omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
              break;
            }
          }
          std::set<StateId> observedIds;
          estimator_.getObservedIds(id, observedIds);
          (*publicationData.updatedStates)[id]
            = State{T_WS,
                    speedAndBias.head<3>(),
                    speedAndBias.segment<3>(3),
                    speedAndBias.tail<3>(),
                    omega_S,
                    timestamp,
                    id,
                    imuMeasurements,
                    estimator_.isKeyframe(id),
                    AlignedMap<uint64_t, kinematics::Transformation>(),
                    observedIds, false,
                    false, AlignedMap<uint64_t, kinematics::Transformation>(),
                    kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
        }
        affectedStates_.clear();

        publicationData.landmarksPublish.reset(new MapPointVector());
        MapPoints landmarks;
        estimator_.getLandmarks(landmarks);
        publicationData.landmarksPublish->reserve(landmarks.size());
        for (const auto &lm : landmarks) {
          publicationData.landmarksPublish->push_back(
            MapPoint(lm.first.value(), lm.second.point, lm.second.quality));
        }

        optimisedGraphCallback_(publicationData.state,
                                publicationData.trackingState,
                                publicationData.updatedStates,
                                publicationData.landmarksPublish);
      }
    }
  }

  shutdown_ = true;
  if(visualisationThread_.joinable()) {
    visualisationThread_.join();
  }
  if(publishingThread_.joinable()) {
    publishingThread_.join();
  }
}

void ThreadedSlam::writeFinalTrajectoryCsv()
{

  stopThreading();

  if(!finalTrajectoryCsvFileName_.empty()) {
    estimator_.writeFinalCsvTrajectory(finalTrajectoryCsvFileName_, rpg_);
  }
}

void ThreadedSlam::writeGlobalTrajectoryCsv(const std::string& csvFileName)
{

  estimator_.writeGlobalCsvTrajectory(csvFileName);
}

void ThreadedSlam::doFinalBa()
{

  AlignmentTerm alignmentTerm;
  while(!submapAlignmentFactorsReceived_.Empty())
  {

    submapAlignmentFactorsReceived_.PopNonBlocking(&alignmentTerm);
    LOG(WARNING) << "Remaining Map-to-Map Contstraints are added between " << alignmentTerm.frame_A_id << " and " << alignmentTerm.frame_B_id;

    if (parameters_.lidar) {
      estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
        alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
        alignmentTerm.sensorError, true, "Tukey");
    }
    else {
      estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
        alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
        alignmentTerm.sensorError, false, "Tukey");
    }

    previousSubmap_ = alignmentTerm.submap_B_ptr;
    previousSubmapId_ = alignmentTerm.frame_B_id;
  }

  stopThreading();

  const int numThreads = parameters_.estimator.realtime_num_threads
      +parameters_.estimator.full_graph_num_threads;
  const bool do_extrinsics_final_ba =
      parameters_.camera.online_calibration.do_extrinsics_final_ba;
  const double sigma_r = do_extrinsics_final_ba ?
                         parameters_.camera.online_calibration.sigma_r_final_ba :
                         0.0;
  const double sigma_alpha = do_extrinsics_final_ba ?
                             parameters_.camera.online_calibration.sigma_alpha_final_ba :
                             0.0;
  std::set<StateId> updatedStatesBa;
  estimator_.doFinalBa(100, posegraphOptimisationSummary_, updatedStatesBa,
      sigma_r, sigma_alpha, numThreads, true, finalBaProgressCallback_);

  StateId currentId = estimator_.currentStateId();
  kinematics::Transformation T_WS = estimator_.pose(currentId);
  SpeedAndBias speedAndBiases = estimator_.speedAndBias(currentId);
  State state;
  state.id = currentId;
  state.T_WS = T_WS;
  state.v_W = speedAndBiases.head<3>();
  state.b_g = speedAndBiases.segment<3>(3);
  state.b_a = speedAndBiases.tail<3>();
  state.omega_S.setZero(); 
  state.timestamp = estimator_.timestamp(currentId);
  state.previousImuMeasurements = imuMeasurementsByFrame_.at(currentId);
  state.isKeyframe = estimator_.isKeyframe(currentId);
  TrackingState trackingState;
  trackingState.id = currentId;
  trackingState.isKeyframe = estimator_.isKeyframe(currentId);
  trackingState.recognisedPlace = estimator_.closedLoop(currentId);
  const double trackingQuality = estimator_.trackingQuality(currentId);
  if(trackingQuality < 0.01) {
    trackingState.trackingQuality = TrackingQuality::Lost;
  } else if (trackingQuality < 0.3){
    trackingState.trackingQuality = TrackingQuality::Marginal;
  } else {
    trackingState.trackingQuality = TrackingQuality::Good;
  }
  trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(currentId, false);
  hasStarted_.store(true);

  if(optimisedGraphCallback_) {

    std::vector<StateId> updatedStateIds;
    PublicationData publicationData;
    publicationData.state = state;
    publicationData.trackingState = trackingState;
    publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
    for(const auto & id : updatedStatesBa) {
      kinematics::Transformation T_WS = estimator_.pose(id);
      SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
      Time timestamp = estimator_.timestamp(id);
      ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
      Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
      for(auto riter = imuMeasurements.rbegin(); riter!=imuMeasurements.rend(); ++riter) {
        if(riter->timeStamp < timestamp) {
          omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
          break;
        }
      }
      std::set<StateId> observedIds;
      estimator_.getObservedIds(id, observedIds);
      (*publicationData.updatedStates)[id] =
          State{T_WS, speedAndBias.head<3>(),speedAndBias.segment<3>(3),
          speedAndBias.tail<3>(),omega_S, timestamp, id, imuMeasurements,
          estimator_.isKeyframe(id),
          AlignedMap<uint64_t, kinematics::Transformation>(), observedIds, true,
          false, AlignedMap<uint64_t, kinematics::Transformation>(),
          kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
    }
    for (const auto &id : affectedStates_) {
      kinematics::Transformation T_WS = estimator_.pose(id);
      SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
      Time timestamp = estimator_.timestamp(id);
      ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
      Eigen::Vector3d omega_S(0.0, 0.0, 0.0); 
      for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
        if (riter->timeStamp < timestamp) {
          omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
          break;
        }
      }
      std::set<StateId> observedIds;
      estimator_.getObservedIds(id, observedIds);
      (*publicationData.updatedStates)[id]
        = State{T_WS,
                speedAndBias.head<3>(),
                speedAndBias.segment<3>(3),
                speedAndBias.tail<3>(),
                omega_S,
                timestamp,
                id,
                imuMeasurements,
                estimator_.isKeyframe(id),
                AlignedMap<uint64_t, kinematics::Transformation>(),
                observedIds, false,
                false, AlignedMap<uint64_t, kinematics::Transformation>(),
                kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
    }
    affectedStates_.clear();

    publicationData.landmarksPublish.reset(new MapPointVector());
    MapPoints landmarks;
    estimator_.getLandmarks(landmarks);
    publicationData.landmarksPublish->reserve(landmarks.size());
    for(const auto & lm : landmarks) {
      publicationData.landmarksPublish->push_back(
            MapPoint(lm.first.value(), lm.second.point, lm.second.quality));
    }

    optimisedGraphCallback_(publicationData.state, publicationData.trackingState,
                            publicationData.updatedStates, publicationData.landmarksPublish);
  }
}

bool ThreadedSlam::getFinalTrajectoryPositions(
    std::map<uint64_t, Eigen::Vector3f>& positionsById) const
{
  positionsById.clear();
  std::vector<ViSlamBackend::FinalTrajectoryPosition> trajectoryPositions;
  if (!estimator_.getFinalTrajectoryPositions(trajectoryPositions)) {
    return false;
  }
  for (const auto& pose : trajectoryPositions) {
    positionsById.emplace(pose.state_id,
                          Eigen::Vector3f(static_cast<float>(pose.px),
                                          static_cast<float>(pose.py),
                                          static_cast<float>(pose.pz)));
  }
  return true;
}

bool ThreadedSlam::getFinalTrajectoryPoses(
    std::vector<ViSlamBackend::FinalTrajectoryPosition>& poses) const
{
  poses.clear();
  return estimator_.getFinalTrajectoryPositions(poses);
}

bool ThreadedSlam::saveMap() {

  stopThreading();

  if(!finalTrajectoryCsvFileName_.empty()) {
    return estimator_.saveMap(mapCsvFileName_);
  }
  return false;
}

void ThreadedSlam::dumpGpsResiduals(const std::string &gpsResCsvFileName)
{

  stopThreading();

  estimator_.dumpGpsResiduals(gpsResCsvFileName);

}

void ThreadedSlam::computeLiveDepthMeasurements(
  const kinematics::Transformation& T_WS_live,
  const okvis::Time& curTime,
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& livePoints,
  std::vector<float>& liveSigmas) {

  livePoints.clear();
  liveSigmas.clear();

  if (previousSubmap_ && !depthMeasurementDeque_.empty()) {

    const kinematics::Transformation T_WD = T_WS_live * T_SD_;
    const kinematics::Transformation T_WM = estimator_.pose(StateId(previousSubmapId_));
    const kinematics::Transformation T_MD = T_WM.inverse() * T_WD;

    static thread_local cv::Mat lowDepthScratch;
    for (size_t i = 0; i < depthMeasurementDeque_.size(); i ++) {
      CameraMeasurement liveDepth = depthMeasurementDeque_[i];

      cv::Mat lowDepth;
      const bool mapping_input_preprocessed =
          parameters_.mappingInputPreprocessor
          && parameters_.mappingInputPreprocessor->enabled()
          && liveDepth.measurement.depthGeometryRevision
                 == parameters_.mappingInputPreprocessor->geometryRevision(
                        static_cast<size_t>(submapConfig_.depthImageResDownsampling));
      if (!mapping_input_preprocessed
          && submapConfig_.depthImageResDownsampling > 1) {
        downsampleDepth(
            liveDepth.measurement.depthImage,
            submapConfig_.depthImageResDownsampling,
            cv::Size(depthCamera_->model.imageWidth(), depthCamera_->model.imageHeight()),
            submapConfig_.near_plane,
            submapConfig_.far_plane,
            lowDepthScratch);
        lowDepth = lowDepthScratch;
      }
      else {
        lowDepth = liveDepth.measurement.depthImage;
      }

      int depthWidth = lowDepth.cols;
      int depthHeight = lowDepth.rows;

      kinematics::Transformation T_delta(Eigen::Matrix4d::Identity());
      if (depthMeasurementDeque_[i].timeStamp != curTime) {

        okvis::State stateMeas;
        if (!trajectory_.getState(depthMeasurementDeque_[i].timeStamp, stateMeas)) {
          okvis::State latestTrajState;
          const std::set<StateId> allIds = trajectory_.stateIds();
          if (!allIds.empty()) {
            StateId latestTrajId = *allIds.rbegin(); 
            trajectory_.getState(latestTrajId, latestTrajState);
            double deltaTime = (curTime-latestTrajState.timestamp).toSec();
            double deltaTime0 = (depthMeasurementDeque_[i].timeStamp-latestTrajState.timestamp).toSec();
            float interpWeight = deltaTime0/deltaTime;
            if (interpWeight > 0 && interpWeight < 1) {
              Eigen::Vector3d interpPos = (1-interpWeight)*latestTrajState.T_WS.r() + interpWeight*T_WS_live.r();
              Eigen::Quaterniond interpQuat = latestTrajState.T_WS.q().slerp(interpWeight, T_WS_live.q());
              okvis::kinematics::Transformation T_interp(interpPos, interpQuat);
              stateMeas.T_WS = T_interp;
            }
          }
        }
        T_delta = T_WD.inverse() * (stateMeas.T_WS * T_SD_);
      }

      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPoints;
      std::vector<float> observedSigmas;

      for (int x = 0; x < depthWidth; x+=10) {
        for (int y = 0; y < depthHeight; y+=10) {
          Eigen::Vector2f imagePoint_xy(static_cast<float>(x), static_cast<float>(y));
          Eigen::Vector3f cameraPoint;
          depthCamera_->model.backProject(imagePoint_xy, &cameraPoint);

          float pointZ = lowDepth.at<float>(y,x);
          float sigmaZ = submapConfig_.sensorError;

          if (!std::isfinite(pointZ) || !std::isfinite(sigmaZ)
              || pointZ < submapConfig_.near_plane || pointZ > submapConfig_.far_plane
              || sigmaZ > 3.0) continue;

          cameraPoint = cameraPoint * pointZ;

          Eigen::Vector3f p_M = (T_MD*T_delta).T3x4().cast<float>() * cameraPoint.homogeneous();
          std::optional<float> occ = okvis::interpFieldMeanOccup<se::Safe::On>(*previousSubmap_, p_M);
          std::optional<Eigen::Vector3f> gradf = okvis::gradFieldMeanOccup<se::Safe::On>(*previousSubmap_, p_M);
          const float occ_margin = 2.0f;
          if (occ && gradf) {
            if (occ.value() > -occ_margin && occ.value() < occ_margin && gradf.value().norm() >= 1e-03) {
              Eigen::Vector3f p_S = (T_SD_*T_delta).T3x4().cast<float>() * cameraPoint.homogeneous();
              observedPoints.push_back(p_S);
              observedSigmas.push_back(sigmaZ);
            }
          } 
        }
      }

      if (observedPoints.size() > submapConfig_.numSubmapFactors) {
        okvis::downsamplePointsUncertainty(observedPoints, observedSigmas, 
                                    livePoints, liveSigmas,
                                    submapConfig_.numSubmapFactors);
      } 
      else {
        livePoints = observedPoints;
        liveSigmas = observedSigmas;
      }     
    }
  }
}

void ThreadedSlam::saveAlignedPoints(size_t id,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points,
                                     std::string type) {
  kinematics::Transformation T_WS = estimator_.pose(StateId(id));
  size_t Npoints = points.size();
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = Npoints;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.resize (cloud.width * cloud.height);
  size_t tmpCnt = 0;
  for (auto& point: cloud) {
    Eigen::Vector3d point_S = points[tmpCnt].cast<double>();
    Eigen::Vector3d point_W = (T_WS.T() * point_S.homogeneous()).head<3>();
    point.x = point_W(0);
    point.y = point_W(1);
    point.z = point_W(2);
    tmpCnt ++;
  }

  std::string mesh_numbering;
  if (id < 10) {
    mesh_numbering = "00000" + std::to_string(id);
  }
  else if (id < 100) {
    mesh_numbering = "0000" + std::to_string(id);
  }
  else if (id < 1000) {
    mesh_numbering = "000" + std::to_string(id);
  }
  else if (id < 10000) {
    mesh_numbering = "00" + std::to_string(id);
  }
  else if (id < 100000) {
    mesh_numbering = "0" + std::to_string(id);
  }
  else if (id < 1000000) {
    mesh_numbering = std::to_string(id);
  }

  std::string saveName = finalTrajectoryCsvFileName_.substr(0, finalTrajectoryCsvFileName_.find("okvis2"))
    + type + '_' + mesh_numbering + ".ply";
  pcl::io::savePLYFileASCII (saveName, cloud);
}

void ThreadedSlam::writeDebugStatisticsCsv(const std::string& csvFilePrefix)
{

  estimator_.writeLidarDebugStatisticsCsv(csvFilePrefix);

}

}  
