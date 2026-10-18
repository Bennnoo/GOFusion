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
 * @file XDatasetReader.hpp
 * @brief Header file for the XDatasetReader class. 99% taken of DatasetReader.hpp
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_XDATASETREADER_HPP_
#define INCLUDE_OKVIS_XDATASETREADER_HPP_

#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <map>
#include <cstdint>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ViSensorBase.hpp>

namespace okvis {

class XDatasetReader : public DatasetReaderBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

        XDatasetReader() = delete;

        XDatasetReader(const std::string& path, const Duration & deltaT = Duration(0.0),
                           ViParameters parameters = ViParameters(), bool use_lidar = false,
                           bool use_depth = false, bool use_rgb = false);

        virtual ~XDatasetReader();

        virtual bool setDatasetPath(const std::string & path) final;

        virtual bool setStartingDelay(const okvis::Duration & deltaT) final;

        virtual bool startStreaming() final;

        virtual bool stopStreaming() final;

        virtual bool isStreaming() final;

        virtual double completion() const final;

        uint64_t inputTickCount() const {
          return static_cast<uint64_t>(counter_.load(std::memory_order_relaxed));
        }

        typedef std::function<
                bool(const okvis::Time &,
                     const Eigen::Vector3d & )> LidarCallback;

        virtual void setLidarCallback(const LidarCallback & lidarCallback) final {
          lidarCallback_ = lidarCallback;
        }

        void setPlaybackRateControl(bool enabled, double target_fps);

    private:

        void processing();
        std::thread processingThread_; 

        std::string path_; 

        std::atomic_bool streaming_; 
        std::atomic_int counter_; 
        size_t numImages_ = 0; 

        Duration deltaT_ = okvis::Duration(0.0); 

        std::ifstream imuFile_; 
        std::ifstream gpsFile_; 
        std::ifstream lidarFile_; 
        std::atomic_bool gpsFlag_; 
        std::string gpsDataType_; 
        std::atomic_bool lidarFlag_; 

        std::atomic_bool depthFlag_; 
        std::atomic_bool rgbFlag_; 
        std::atomic_bool imuFlag_; 
        LidarCallback lidarCallback_; 

        std::vector < std::vector < std::pair<std::string, std::string> > > allImageNames_;
        std::vector < std::pair<std::string, std::string> > allDepthNames_;
        std::map<uint64_t, std::string> depthRgbNames_;
        bool pairDepthWithRgb_ = false;
        bool playbackRateControlEnabled_ = false;
        double playbackTargetFps_ = 60.0;
        std::shared_ptr<const MappingImagePreprocessor> mappingInputPreprocessor_;
        size_t mappingInputDownsampling_ = 1;
        std::string mappingRgbStream_ = "rgb0";
        std::string mappingDepthStream_ = "depth0";

        int numCameras_ = -1;
        size_t depthCameraId_ = 0;
        okvis::Time t_gps_; 

    };

} 

#endif 
