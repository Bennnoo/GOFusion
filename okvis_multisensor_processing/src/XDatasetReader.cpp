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
 * @file XDatasetReader.cpp
 * @brief Source file for the XDatasetReader class. 99% taken of DatasetReader.cpp
 * @author Simon Boche
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <okvis/ViInterface.hpp>
#include <okvis/XDatasetReader.hpp>

namespace okvis {

XDatasetReader::XDatasetReader(const std::string & path,
                             const Duration & deltaT, ViParameters parameters, bool use_lidar,
                             bool use_depth, bool use_rgb) :
        deltaT_(deltaT), lidarFlag_(use_lidar), depthFlag_(use_depth), rgbFlag_(use_rgb), imuFlag_(parameters.imu.use){

  mappingInputPreprocessor_ = parameters.mappingInputPreprocessor;
  mappingInputDownsampling_ = parameters.mappingInputDownsampling;
  if (mappingInputPreprocessor_) {
    mappingRgbStream_ = mappingInputPreprocessor_->rgbStream();
    mappingDepthStream_ = mappingInputPreprocessor_->depthStream();
  }

  if(parameters.gps) {
    gpsFlag_ = true;
    gpsDataType_ = (*parameters.gps).type;
    OKVIS_ASSERT_TRUE(Exception, gpsDataType_=="cartesian" || gpsDataType_=="geodetic" || gpsDataType_=="geodetic-leica",
                      "Unknown GPS data type specified")
  }
  else {
    gpsFlag_ = false;
  }

  numCameras_ = parameters.nCameraSystem.numCameras();
  streaming_ = false;
  setDatasetPath(path);
  counter_ = 0;
  t_gps_ = okvis::Time(0.0);

  if(use_depth) {
    if (mappingInputPreprocessor_ && mappingInputPreprocessor_->enabled()) {
      depthCameraId_ = mappingInputPreprocessor_->targetCameraIndex();
    } else {
      for(int camId = 0; camId < numCameras_; camId ++) {
        if(parameters.nCameraSystem.isDepthCamera(camId)) {
          depthCameraId_ = camId;
        }
      }
    }
  }
}

XDatasetReader::~XDatasetReader() {
  stopStreaming();
}

bool XDatasetReader::setDatasetPath(const std::string & path) {
  path_ = path;
  return true;
}

bool XDatasetReader::setStartingDelay(const Duration &deltaT)
{
  if(streaming_) {
    LOG(WARNING)<< "starting delay ignored, because streaming already started";
    return false;
  }
  deltaT_ = deltaT;
  return true;
}

void XDatasetReader::setPlaybackRateControl(const bool enabled, const double target_fps)
{
  if (streaming_) {
    LOG(WARNING) << "playback rate control change ignored, because streaming already started";
    return;
  }
  playbackRateControlEnabled_ = enabled;
  playbackTargetFps_ = target_fps;
}

bool XDatasetReader::isStreaming()
{
  return streaming_;
}

double XDatasetReader::completion() const {
  if(streaming_) {
    return double(counter_)/double(numImages_);
  }
  return 0.0;
}

bool XDatasetReader::startStreaming() {
  OKVIS_ASSERT_TRUE(Exception, !imagesCallbacks_.empty(), "no add image callback registered")
  if(imuFlag_)
    OKVIS_ASSERT_TRUE(Exception, !imuCallbacks_.empty(), "no add IMU callback registered")
  if(gpsFlag_)
    OKVIS_ASSERT_TRUE(Exception, geodeticGpsCallback_ || gpsCallback_, "no add GPS callback registered")
  if(lidarFlag_)
    OKVIS_ASSERT_TRUE(Exception, lidarCallback_, "no add Lidar callback registered")

  std::string line;
  if(imuFlag_) {
    imuFile_.open(path_ + "/imu0/data.csv");
    OKVIS_ASSERT_TRUE(Exception, imuFile_.good(), "no imu file found at " << path_+"/imu0/data.csv");
    int number_of_lines = 0;
    while (std::getline(imuFile_, line))
      ++number_of_lines;
    LOG(INFO)<< "No. IMU measurements: " << number_of_lines-1;
    if (number_of_lines - 1 <= 0) {
      LOG(ERROR)<< "no imu messages present in " << path_+"/imu0/data.csv";
      return -1;
    }

    imuFile_.clear();
    imuFile_.seekg(0, std::ios::beg);
    std::getline(imuFile_, line);
  }

  if(gpsFlag_)
  {

    std::string gline;
    if(gpsDataType_ =="cartesian")
      gpsFile_.open(path_ + "/gps0/data.csv");
    else if(gpsDataType_=="geodetic")
      gpsFile_.open(path_ + "/gps0/data_raw.csv");
    else if(gpsDataType_ == "geodetic-leica")
      gpsFile_.open(path_+"gnss.csv");
    OKVIS_ASSERT_TRUE(Exception, gpsFile_.good(), "no gps file found at " << path_+"/gps0/");
    int gnumber_of_lines = 0;
    while (std::getline(gpsFile_, gline))
      ++gnumber_of_lines;
    LOG(INFO)<< "No. GPS measurements: " << gnumber_of_lines-1;
    if (gnumber_of_lines - 1 <= 0) {
      LOG(ERROR)<< "no gps messages present in " << path_+"/gps0/data.csv";
      return -1;
    }

    gpsFile_.clear();
    gpsFile_.seekg(0, std::ios::beg);
    std::getline(gpsFile_, gline);

  }

  if(lidarFlag_){

    std::string lidarLine;
    lidarFile_.open(path_ + "/lidar0/data.csv");
    OKVIS_ASSERT_TRUE(Exception, lidarFile_.good(), "no lidar file found at " << path_+"/lidar0/data.csv");

    std::getline(lidarFile_, lidarLine);
  }

  if(depthFlag_){

    int num_depth_images = 0;
    allDepthNames_.clear();
    depthRgbNames_.clear();
    pairDepthWithRgb_ = false;
    std::ifstream depthDataFile(path_ + "/" + mappingDepthStream_ + "/data.csv");
    OKVIS_ASSERT_TRUE(Exception, depthDataFile.good(),
                      "no mapping depth index found at " << path_ << "/"
                                                         << mappingDepthStream_
                                                         << "/data.csv")
    std::pair<std::string, std::string> depthNames;
    std::getline(depthDataFile, line);
    while (std::getline(depthDataFile, line)) {
      ++num_depth_images;
      std::stringstream stream(line);
      std::string s0, s1;
      std::getline(stream, s0, ',');

      std::getline(stream, s1, ',');

      while (!s1.empty() && (s1.back() == '\r' || s1.back() == ',' || s1.back() == ' ' || s1.back() == '\t')) {
        s1.pop_back();
      }

      while (!s1.empty() && (s1.front() == ' ' || s1.front() == '\t')) {
        s1.erase(0, 1);
      }

      depthNames = std::make_pair(s0, s1);
      allDepthNames_.push_back(depthNames);
    }
    LOG(INFO) << "No. depth images: " << num_depth_images;

    std::ifstream rgbPairDataFile(path_ + "/" + mappingRgbStream_ + "/data.csv");
    if (rgbPairDataFile.good()) {
      int num_rgb_images = 0;
      std::getline(rgbPairDataFile, line);
      while (std::getline(rgbPairDataFile, line)) {
        std::stringstream stream(line);
        std::string s0, s1;
        std::getline(stream, s0, ',');
        std::getline(stream, s1);

        while (!s1.empty() && (s1.back() == '\r' || s1.back() == ',' || s1.back() == ' ' || s1.back() == '\t')) {
          s1.pop_back();
        }
        while (!s1.empty() && (s1.front() == ' ' || s1.front() == '\t')) {
          s1.erase(0, 1);
        }

        while (!s0.empty() && (s0.back() == '\r' || s0.back() == ',' || s0.back() == ' ' || s0.back() == '\t')) {
          s0.pop_back();
        }
        while (!s0.empty() && (s0.front() == ' ' || s0.front() == '\t')) {
          s0.erase(0, 1);
        }

        if (!s0.empty() && !s1.empty()) {
          depthRgbNames_[std::stoull(s0)] = s1;
          ++num_rgb_images;
        }
      }
      pairDepthWithRgb_ = !depthRgbNames_.empty();
      LOG(INFO) << "No. rgb images for depth pairing: " << num_rgb_images;
    } else {
      LOG(INFO) << "No " << mappingRgbStream_
                << "/data.csv found. Depth callback remains depth-only.";
    }
  }

  std::vector<okvis::Time> times;
  okvis::Time latest(0);
  int num_camera_images = 0;
  int i = 0;
  while (i<numCameras_) {
    std::ifstream camDataFile(path_ + "/cam" + std::to_string(i) + "/data.csv");
    if(!camDataFile.good()) {
      OKVIS_ASSERT_TRUE(Exception, i>0, "No camera data found");
      break;
    }
    num_camera_images = 0;
    std::vector < std::pair<std::string, std::string> > imageNames;
    std::getline(camDataFile, line);
    while (std::getline(camDataFile, line)) {
      ++num_camera_images;
      std::stringstream stream(line);
      std::string s0, s1;
      std::getline(stream, s0, ',');
      std::getline(stream, s1, ',');

      if (s1.back() == '\r') {
        s1 = s1.substr(0, s1.size() - 1);
      }
      imageNames.push_back(std::make_pair(s0,s1));
    }
    allImageNames_.push_back(imageNames);
    LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
    if(i==0) {
      numImages_ = num_camera_images;
    }
    ++i; 
  }

  if(rgbFlag_) {

    int num_rgb_images = 0;
    std::ifstream rgbDataFile(path_ + "/" + mappingRgbStream_ + "/data.csv");
    OKVIS_ASSERT_TRUE(Exception, rgbDataFile.good(),
                      "no mapping RGB index found at " << path_ << "/"
                                                       << mappingRgbStream_
                                                       << "/data.csv")
    std::vector < std::pair<std::string, std::string> > rgbNames;
    std::getline(rgbDataFile, line);
    while (std::getline(rgbDataFile, line)) {
      ++num_rgb_images;
      std::stringstream stream(line);
      std::string s0, s1;
      std::getline(stream, s0, ',');
      std::getline(stream, s1);

      if (s1.back() == '\r') {
        s1 = s1.substr(0, s1.size() - 1);
      }
      rgbNames.push_back(std::make_pair(s0,s1));
    }
    LOG(INFO) << "No. rgb images: " << num_rgb_images;
    allImageNames_.push_back(rgbNames);
  }

  counter_ = 0;
  streaming_ = true;
  processingThread_ = std::thread(&XDatasetReader::processing, this);

  return true;
}

bool XDatasetReader::stopStreaming() {

  if(processingThread_.joinable()) {
    processingThread_.join();
    streaming_ = false;
  }
  return true;
}

void  XDatasetReader::processing() {
  std::string line;
  okvis::Time start(0.0);
  const size_t numCameras = allImageNames_.size();
  std::vector < std::vector < std::pair<std::string, std::string> > ::iterator
  > cam_iterators(numCameras);
  for (size_t i = 0; i < numCameras; ++i) {
    cam_iterators.at(i) = allImageNames_.at(i).begin();
  }

  std::vector < std::pair<std::string, std::string> > ::iterator depth_iterators;
  depth_iterators = allDepthNames_.begin();

  const bool useRateControl =
      playbackRateControlEnabled_ && (playbackTargetFps_ > 0.0);
  const auto pacingPeriod = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / std::max(playbackTargetFps_, 1e-6)));
  const int64_t pacingPeriodNs = std::max<int64_t>(
      1,
      std::chrono::duration_cast<std::chrono::nanoseconds>(pacingPeriod).count());
  bool pacingInitialized = false;
  std::chrono::steady_clock::time_point nextDispatchWallTime;
  size_t skippedCameraFrames = 0;
  size_t skippedDepthFrames = 0;

  const uint64_t tolNSec = 10000000; 
  bool loggedDepthRgbPairing = false;
  while (streaming_) {
    size_t cameraFramesToSkipBeforeProcessing = 0;
    if (useRateControl) {
      const auto now = std::chrono::steady_clock::now();
      if (!pacingInitialized) {
        nextDispatchWallTime = now;
        pacingInitialized = true;
      }
      if (nextDispatchWallTime > now) {
        std::this_thread::sleep_until(nextDispatchWallTime);
      } else {
        const int64_t lagNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - nextDispatchWallTime)
                .count();
        if (lagNs > pacingPeriodNs) {
          cameraFramesToSkipBeforeProcessing =
              static_cast<size_t>(lagNs / pacingPeriodNs);
          nextDispatchWallTime += pacingPeriod
                                  * static_cast<int64_t>(
                                      cameraFramesToSkipBeforeProcessing);
        }
      }
    }

    bool stopStreamingAfterCurrentFrame = false;

    if (cameraFramesToSkipBeforeProcessing > 0) {
      for (size_t n = 0; n < cameraFramesToSkipBeforeProcessing; ++n) {
        for (size_t i = 0; i < numCameras; ++i) {
          if (cam_iterators.at(i) == allImageNames_.at(i).end()) {
            streaming_ = false;
            return;
          }
          ++cam_iterators.at(i);
        }
        ++counter_;
        ++skippedCameraFrames;
      }
      if (skippedCameraFrames > 0 && skippedCameraFrames % 200 == 0) {
        LOG(INFO) << "Realtime input: skipped " << skippedCameraFrames
                  << " camera frames to keep pacing.";
      }
    }

    bool synched = false;
    while(!synched) {
      uint64_t max_timestamp = 0;
      for (size_t i = 0; i < numCameras; ++i) {
        if(cam_iterators.at(i) == allImageNames_.at(i).end()) {
          streaming_ = false;
          return; 
        }
        uint64_t tcam = std::atol(cam_iterators.at(i)->first.c_str());
        if(tcam>max_timestamp) {
          max_timestamp = tcam;
        }
      }
      synched=true;
      for (size_t i = 0; i < numCameras; ++i) {
        uint64_t tcam = std::atol(cam_iterators.at(i)->first.c_str());
        const uint64_t dt = (max_timestamp > tcam) ? (max_timestamp - tcam) : 0;
        if(dt > tolNSec) {
          LOG(INFO) << "Out of sync by " << dt;
          cam_iterators.at(i)++; 
          synched = false;
        }
      }
    }

    for (size_t i = 0; i < numCameras; ++i) {
      if(i>0) {
        int64_t tcam = std::atol(cam_iterators.at(0)->first.c_str());
        int64_t tcam_i = std::atol(cam_iterators.at(i)->first.c_str());
        OKVIS_ASSERT_TRUE(Exception, abs(tcam-tcam_i)<int64_t(tolNSec),
                          "timestamp mismatch " << double(abs(tcam-tcam_i))/1.0e6 << " ms at " << tcam_i);
      }
    }

    okvis::Time t;
    std::map<size_t, cv::Mat> images;
    std::map<size_t, cv::Mat> depthImages;

    bool missingImageInFrame = false;

    cv::Mat mappingRgbForFrame;
    uint64_t mappingRgbForFrameTimestamp = 0;
    std::string mappingRgbForFrameFilename;
    if (rgbFlag_ && numCameras > 0) {
      mappingRgbForFrameTimestamp =
          std::stoull(cam_iterators.back()->first);
      mappingRgbForFrameFilename = cam_iterators.back()->second;
      const std::string mapping_rgb_path =
          path_ + "/" + mappingRgbStream_ + "/data/"
          + mappingRgbForFrameFilename;
      mappingRgbForFrame = cv::imread(mapping_rgb_path, cv::IMREAD_COLOR);
      if (mappingRgbForFrame.empty()) {
        LOG(WARNING) << "Missing RGB image (stop after current frame): "
                     << mapping_rgb_path;
        missingImageInFrame = true;
        stopStreamingAfterCurrentFrame = true;
      } else {
        cv::cvtColor(mappingRgbForFrame, mappingRgbForFrame,
                     cv::COLOR_BGR2RGB);
      }
    }

    if (missingImageInFrame) {
      streaming_ = false;
      return;
    }

    for (size_t i = 0; i < numCameras; ++i) {
      cv::Mat filtered;
      std::pair<okvis::Time, cv::Mat> timestamp_filtered;
      std::string filename;
      if(rgbFlag_ && i == (numCameras - 1)) {
        filename = path_ + "/" + mappingRgbStream_ + "/data/"
                   + cam_iterators.at(i)->second;
        filtered = mappingRgbForFrame;
      } else {
        filename = path_ + "/cam" + std::to_string(i) + "/data/" + cam_iterators.at(i)->second;
        filtered = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (filtered.empty()) {
          LOG(WARNING) << "Missing camera image (stop after current frame): " << filename;
          missingImageInFrame = true;
          stopStreamingAfterCurrentFrame = true;
          break;
        }
      }

      t.fromNSec(std::atol(cam_iterators.at(i)->first.c_str()));
      timestamp_filtered = std::make_pair(t, filtered);

      if (start == okvis::Time(0.0)) {
        start = t;
      }

      if(lidarFlag_){

        okvis::Time t_lidar = start;
        std::string lidarLine;
        do {
          if (!std::getline(lidarFile_, lidarLine)) {
            streaming_ = false;
            return;
          }

          std::stringstream lidarStream(lidarLine);
          std::string sl;

          std::getline(lidarStream, sl, ',');
          uint64_t lidarNanoseconds = std::stol(sl.c_str());

          Eigen::Vector3d ptXYZ;
          for(int j = 0; j < 3; ++j){
            std::getline(lidarStream, sl, ',');
            ptXYZ[j] = std::stod(sl);
          }

          std::getline(lidarStream, sl);

          t_lidar.fromNSec(lidarNanoseconds);

          if (t_lidar - start + okvis::Duration(1.0) > deltaT_) {
            lidarCallback_(t_lidar, ptXYZ);
          }

        } while (t_lidar <= t);
      }

      if (depthFlag_ && !stopStreamingAfterCurrentFrame) {
        auto dispatchDepth = [&](const std::pair<std::string, std::string>& depth_entry,
                                 const uint64_t depthNanoseconds) -> bool {
          okvis::Time t_depth;
          t_depth.fromNSec(depthNanoseconds);

          std::string depth_filename = path_ + "/" + mappingDepthStream_
                                       + "/data/" + depth_entry.second;
          const cv::Mat encoded_depth =
              cv::imread(depth_filename, cv::IMREAD_UNCHANGED);
          if (encoded_depth.empty()) {
            LOG(WARNING) << "Depth file missing (skip): " << depth_filename;
            return false;
          }

          cv::Mat depth;
          encoded_depth.convertTo(depth, CV_32F);

          if (t_depth - start >= deltaT_) {
            okvis::CameraMeasurement camMeasurement;
            camMeasurement.timeStamp = t_depth;
            const bool remap_encoded_depth =
                mappingInputPreprocessor_
                && mappingInputPreprocessor_->enabled()
                && encoded_depth.channels() == 1;
            camMeasurement.measurement.depthImage =
                (remap_encoded_depth)
                    ? encoded_depth
                    : depth;

            if (pairDepthWithRgb_) {

              auto it_after = depthRgbNames_.lower_bound(depthNanoseconds);
              auto best_it = depthRgbNames_.end();
              uint64_t best_dt = tolNSec + 1;

              if (it_after != depthRgbNames_.end()) {
                const uint64_t dt_after = (it_after->first >= depthNanoseconds)
                                            ? (it_after->first - depthNanoseconds)
                                            : (depthNanoseconds - it_after->first);
                if (dt_after < best_dt) {
                  best_dt = dt_after;
                  best_it = it_after;
                }
              }

              if (it_after != depthRgbNames_.begin()) {
                auto it_before = it_after;
                --it_before;
                const uint64_t dt_before = (it_before->first >= depthNanoseconds)
                                             ? (it_before->first - depthNanoseconds)
                                             : (depthNanoseconds - it_before->first);
                if (dt_before < best_dt) {
                  best_dt = dt_before;
                  best_it = it_before;
                }
              }

              if (best_it != depthRgbNames_.end() && best_dt <= tolNSec) {
                cv::Mat rgb;
                if (best_it->first == mappingRgbForFrameTimestamp
                    && best_it->second == mappingRgbForFrameFilename
                    && !mappingRgbForFrame.empty()) {
                  rgb = mappingRgbForFrame;
                } else {
                  const std::string rgbFilename = path_ + "/"
                                                  + mappingRgbStream_
                                                  + "/data/"
                                                  + best_it->second;
                  rgb = cv::imread(rgbFilename, cv::IMREAD_COLOR);
                  if (!rgb.empty()) {
                    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
                  }
                }
                if (!rgb.empty()) {
                  camMeasurement.measurement.image = rgb;

                  if (!loggedDepthRgbPairing) {
                    LOG(INFO) << "Depth callback carries paired " << mappingRgbStream_
                              << " images for mapping camera id " << depthCameraId_;
                    loggedDepthRgbPairing = true;
                  }
                }
              }
            }

            {
              depthImages[depthCameraId_] = depth;

            }

            if (depthCallback_) {

              if (mappingInputPreprocessor_
                  && mappingInputPreprocessor_->enabled()) {
                OKVIS_ASSERT_TRUE(
                    Exception,
                    mappingInputPreprocessor_->process(
                        depthCameraId_, camMeasurement,
                        mappingInputDownsampling_),
                    "mapping input target camera does not match the configured depth camera")
              }
              if (camMeasurement.measurement.depthImage.type() != CV_32FC1) {
                cv::Mat mapping_depth_float;
                camMeasurement.measurement.depthImage.convertTo(
                    mapping_depth_float, CV_32F);
                camMeasurement.measurement.depthImage =
                    std::move(mapping_depth_float);
              }
              std::map<size_t, std::vector<okvis::CameraMeasurement>> si_data;
              si_data[depthCameraId_] = {camMeasurement};
              depthCallback_(si_data);
            }
          }
          return true;
        };

        if (useRateControl) {
          bool haveLatestDepth = false;
          std::pair<std::string, std::string> latestDepthEntry;
          uint64_t latestDepthNanoseconds = 0;
          size_t consumedDepthEntries = 0;
          while (depth_iterators != allDepthNames_.end()) {
            const uint64_t depthNanoseconds = std::stoull(depth_iterators->first);
            okvis::Time t_depth;
            t_depth.fromNSec(depthNanoseconds);
            if (t_depth > t) {
              break;
            }
            latestDepthEntry = *depth_iterators;
            latestDepthNanoseconds = depthNanoseconds;
            ++depth_iterators;
            ++consumedDepthEntries;
            haveLatestDepth = true;
          }

          if (depth_iterators == allDepthNames_.end()) {
            LOG(INFO) << "depth stop streaming";
            stopStreamingAfterCurrentFrame = true;
          }

          if (consumedDepthEntries > 1) {
            skippedDepthFrames += (consumedDepthEntries - 1);
            if (skippedDepthFrames > 0 && skippedDepthFrames % 200 == 0) {
              LOG(INFO) << "Realtime input: skipped " << skippedDepthFrames
                        << " stale depth frames.";
            }
          }

          if (haveLatestDepth) {
            dispatchDepth(latestDepthEntry, latestDepthNanoseconds);
          }
        } else {

          okvis::Time t_depth = start;
          while(true) {
            if (depth_iterators == allDepthNames_.end()) {
              LOG(INFO) << "depth stop streaming";
              stopStreamingAfterCurrentFrame = true;
              break;
            }
            const uint64_t depthNanoseconds = std::stoull(depth_iterators->first);
            t_depth.fromNSec(depthNanoseconds);
            if (t_depth > t) {
              break;
            }
            dispatchDepth(*depth_iterators, depthNanoseconds);
            ++depth_iterators;
          }
        }
      }

      if(imuFlag_){
        okvis::Time t_imu = start;
        do {
          if (!std::getline(imuFile_, line)) {
            streaming_ = false;
            return;
          }

          std::stringstream stream(line);
          std::string s;
          std::getline(stream, s, ',');
          uint64_t nanoseconds = std::stol(s.c_str());

          Eigen::Vector3d gyr;
          for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            gyr[j] = std::stof(s);
          }

          Eigen::Vector3d acc;
          for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            acc[j] = std::stof(s);
          }

          t_imu.fromNSec(nanoseconds);

          if (t_imu - start + okvis::Duration(1.0) > deltaT_) {
            for (auto &imuCallback : imuCallbacks_) {
              imuCallback(t_imu, acc, gyr);
            }
          }

        } while (t_imu <= t);
      }

      if(gpsFlag_){

        std::string gline;
        while(t_gps_ <= t) {
          if (!std::getline(gpsFile_, gline)) {
            streaming_ = false;
            return;
          }

          std::stringstream gstream(gline);
          std::string gs;

          if(gpsDataType_ == "cartesian"){

            std::getline(gstream, gs, ',');
            uint64_t gnanoseconds = std::stol(gs.c_str()) - GNSS_LEAP_NANOSECONDS;

            Eigen::Vector3d pos;
            for (int j = 0; j < 3; ++j) {
              std::getline(gstream, gs, ',');
              pos[j] = std::stof(gs);
            }

            Eigen::Vector3d err;
            for (int j = 0; j < 3; ++j) {
              std::getline(gstream, gs, ',');
              err[j] = std::stof(gs);
            }

            t_gps_.fromNSec(gnanoseconds);

            if (t_gps_ - start + okvis::Duration(1.0) > deltaT_) {
              gpsCallback_(t_gps_, pos, err);
            }

          }
          else if (gpsDataType_ == "geodetic"){

            std::getline(gstream, gs, ',');
            uint64_t gnanoseconds = std::stol(gs.c_str()) - GNSS_LEAP_NANOSECONDS;

            std::getline(gstream, gs, ',');
            double lat = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double lon = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double alt = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double hErr = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double vErr = std::stod(gs.c_str());

            t_gps_.fromNSec(gnanoseconds);

            if (t_gps_ - start + okvis::Duration(1.0) > deltaT_) {
              geodeticGpsCallback_(t_gps_, lat, lon, alt, hErr, vErr);
            }
          }
          else if (gpsDataType_ == "geodetic-leica"){

            std::getline(gstream, gs, ',');
            uint64_t gnanoseconds = std::stol(gs.c_str()) - GNSS_LEAP_NANOSECONDS;

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');
            double lat = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double lon = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double alt = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');

            std::getline(gstream, gs, ',');
            double hErr = std::stod(gs.c_str());

            std::getline(gstream, gs, ',');
            double vErr = std::stod(gs.c_str());

            t_gps_.fromNSec(gnanoseconds);

            if (t_gps_ - start + okvis::Duration(1.0) > deltaT_) {
              geodeticGpsCallback_(t_gps_, lat, lon, alt, hErr, vErr);
            }
          }

        } 

      }

      if (t - start >= deltaT_) {
        images[i] = filtered;

      }

      cam_iterators[i]++;

    }
    if (missingImageInFrame) {
      if (useRateControl) {
        nextDispatchWallTime += pacingPeriod;
      }
      if (stopStreamingAfterCurrentFrame) {
        streaming_ = false;
      }
      continue;
    }
    if(images.size() == numCameras) {
      for (auto& imagesCallback : imagesCallbacks_) {
        imagesCallback(t, images, depthImages);
      }

      ++counter_;
    }

    if (useRateControl) {
      nextDispatchWallTime += pacingPeriod;
    }

    if (stopStreamingAfterCurrentFrame) {
      streaming_ = false;
    }
  }

  return;
}

}
