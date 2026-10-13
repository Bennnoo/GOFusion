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
 * @file okvis/Component.hpp
 * @brief Header file for the Component class. Load/save/multisession stuff.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_COMPONENT_HPP_
#define INCLUDE_OKVIS_COMPONENT_HPP_

#include <okvis/ViGraphEstimator.hpp>

namespace okvis {

class Component
{
  public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Component(const ImuParameters &imuParameters,
            const cameras::NCameraSystem& nCameraSystem,
            ViGraphEstimator &fullGraph,
            std::map<StateId, MultiFramePtr> multiFrames);

  Component(const ImuParameters &imuParameters, const cameras::NCameraSystem& nCameraSystem);

  bool load(const std::string &path);

  bool save(const std::string &path);

  ImuParameters imuParameters_; 
  cameras::NCameraSystem nCameraSystem_; 
  ViGraphEstimator *fullGraph_ = nullptr; 
  std::unique_ptr<ViGraphEstimator> fullGraphOwn_; 
  std::map<StateId, MultiFramePtr> multiFrames_; 
};

}  

#endif 
