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
 * @file TrajectoryOutput.hpp
 * @brief Header file for the TrajectoryOutput class.
 * @author Stefan Leutenegger
 */

#ifndef OKVIS_TRAJECTORYOUTPUT_HPP
#define OKVIS_TRAJECTORYOUTPUT_HPP

#include "okvis/QueuedTrajectory.hpp"
#include <fstream>
#include <memory>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/QueuedTrajectory.hpp>

namespace okvis {

class TrajectoryOutput {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  TrajectoryOutput(bool draw = true);

  TrajectoryOutput(const std::string & filename, bool rpg = false, bool draw = true);

  void setCsvFile(const std::string & filename, bool rpg = false);

  void setRGBCsvFile(const std::string & filename);

  void processState(const State& state, const TrackingState & trackingState,
                    std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
                    std::shared_ptr<const okvis::MapPointVector> landmarks);

  bool addImuMeasurement(const okvis::Time& stamp,
                         const Eigen::Vector3d& alpha,
                         const Eigen::Vector3d& omega);

  bool processRGBImage(const okvis::Time &, const cv::Mat &);

  void drawTopView(cv::Mat & outImg);

private:
  std::fstream csvFile_; 
  std::fstream rgbCsvFile_;  
  bool rpg_ = false; 

  bool draw_ = true; 

  typedef std::tuple<
      State, TrackingState, std::shared_ptr<const AlignedMap<StateId, State>>,
      std::shared_ptr<const okvis::MapPointVector>> GraphStates;

  threadsafe::Queue<std::shared_ptr<GraphStates>> states_; 
  Trajectory trajectory_; 
  threadsafe::Queue<ImuMeasurement> imuMeasurements_; 

  cv::Mat _image; 
  int _imageSize = 500; 
  std::vector<cv::Point2d> _path; 
  std::vector<cv::Point2d> _gps; 
  std::vector<double> _heights; 
  double _scale = 1.0; 
  double _min_x = -0.5; 
  double _min_y = -0.5; 
  double _min_z = -0.5; 
  double _max_x = 0.5; 
  double _max_y = 0.5; 
  double _max_z = 0.5; 
  const double _frameScale = 0.2;  

  okvis::QueuedTrajectory<cv::Mat> rgbTrajectory_;  

  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const;

  void drawPath();

  void drawGps();

  void drawPathNoncausal();

  static bool createCsvFile(const std::string &filename, std::fstream& stream,
                            const bool rpg = false);

  static bool writeStateToCsv(std::fstream& csvFile, const okvis::State& state,
                              const bool rpg = false);

  okvis::kinematics::Transformation _T_AW; 

};

}

#endif 
