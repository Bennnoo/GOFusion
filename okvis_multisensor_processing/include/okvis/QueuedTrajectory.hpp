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
 * @file QueuedTrajectory.hpp
 * @brief Header file for the QueuedTrajectory class.
 * @author Simon Schaefer
 */

#ifndef OKVIS_QUEUEDTRAJECTORY_HPP
#define OKVIS_QUEUEDTRAJECTORY_HPP

#include <vector>
#include <deque>
#include <Eigen/Core>

#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ViInterface.hpp>

namespace okvis {

template<class MEASUREMENT_T>
class QueuedTrajectory {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef Measurement<MEASUREMENT_T> MeasurementTyped; 

  std::vector<std::pair<MeasurementTyped, okvis::State>> getStates(okvis::Trajectory& trajectory);

  void enqueue(const MeasurementTyped& measurement);

  void enqueue(const MEASUREMENT_T& data, const okvis::Time& timestamp);

private:

  std::deque<MeasurementTyped, Eigen::aligned_allocator<MeasurementTyped> > measurementQueue_;

};

#include "implementation/QueuedTrajectory.hpp"

}

#endif 
