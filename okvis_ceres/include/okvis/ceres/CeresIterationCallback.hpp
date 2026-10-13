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
 * @file CeresIterationCallback.hpp
 * @brief Header file for the CeresIterationCallback class.
 *        Used to enforce a time limit on the ceres optimization.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_CERESITERATIONCALLBACK_HPP
#define INCLUDE_OKVIS_CERES_CERESITERATIONCALLBACK_HPP

#include <ceres/iteration_callback.h>

namespace okvis {

namespace ceres {

class CeresIterationCallback : public ::ceres::IterationCallback {
 public:

  CeresIterationCallback(double timeLimit, int iterationMinimum);

  virtual inline ~CeresIterationCallback() override {
  }

  ::ceres::CallbackReturnType operator()(
      const ::ceres::IterationSummary& summary) override;

  void setTimeLimit(double timeLimit);

  void setMinimumIterations(int iterationMinimum);

 private:
  double timeLimit_; 
  int iterationMinimum_; 
};

}  
}  

#endif 
