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
 * @file ViVisualizer.hpp
 * @brief Header file for the VioVisualizer class.
 * @author Pascal Gohl
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIOVISUALIZER_HPP_
#define INCLUDE_OKVIS_VIOVISUALIZER_HPP_

#include <okvis/assert_macros.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace okvis {

class ViVisualizer {
 public:

  struct VisualizationData {

    enum class TrackingQuality {
      Good,
      Marginal,
      Lost
    };

    typedef std::shared_ptr<VisualizationData> Ptr;

    okvis::ObservationVector observations;    
    std::shared_ptr<okvis::MultiFrame> currentFrames; 
    okvis::kinematics::Transformation T_WS;  
    AlignedVector<kinematics::Transformation> T_SCi;  
    bool isKeyframe = false; 
    bool recognisedPlace = false; 
    TrackingQuality trackingQuality = TrackingQuality::Good; 
    bool previewOnly = false; 
    size_t stereoDescriptorMatches = 0; 
  };

  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  ViVisualizer(okvis::ViParameters& parameters);

  virtual ~ViVisualizer();

  void init(okvis::ViParameters& parameters);

  cv::Mat drawMatches(VisualizationData::Ptr& data, size_t image_number);

 private:

  cv::Mat drawKeypoints(VisualizationData::Ptr& data, size_t cameraIndex);

  okvis::ViParameters parameters_;
};

} 

#endif 
