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
 * @file ViInterface.hpp
 * @brief Header file for the ViInterface class.
 * @author Paul Furgale
 * @author Stefan Leutenegger
 * @author Andreas Froster
 */

#ifndef INCLUDE_OKVIS_VIINTERFACE_HPP_
#define INCLUDE_OKVIS_VIINTERFACE_HPP_

#include <cstdint>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#pragma GCC diagnostic pop
#include <okvis/assert_macros.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Measurements.hpp>
#include <optional>

namespace okvis {

class Propagator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Propagator(const Time & t_start);

  Propagator(Propagator && ) = default;

  Propagator(const Propagator & ) = default;

  bool appendTo(
      const ImuMeasurementDeque & imuMeasurements,
      const kinematics::Transformation& T_WS,
      const SpeedAndBias & speedAndBiases, const Time & t_end);

  bool getState(
      const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
      kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
      Eigen::Vector3d & omega_S);

protected:

  double sinc(double x);

  Time t_start_; 
  Time t_end_; 
  double Delta_t_ = 0.0; 

  Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); 
  Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); 
  Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); 
  Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); 

  Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); 

  Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); 

  Eigen::Vector3d omega_S_ = Eigen::Vector3d::Zero(); 

  SpeedAndBias sb_ref_ = SpeedAndBias::Zero(); 
};

class BatchedLidarPropagator : Propagator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BatchedLidarPropagator(const Time & t_start, const ImuMeasurementDeque & imuMeasurements);

  bool appendTo(
          const ImuMeasurementDeque & imuMeasurements,
          const kinematics::Transformation& T_WS,
          const SpeedAndBias & speedAndBiases, const Time & t_end);

  bool getState(
          const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
          kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
          Eigen::Vector3d & omega_S);

private:

  ImuMeasurementDeque::const_iterator imu_iterator_;
  bool hasStarted_ = false;

  Eigen::Quaterniond Delta_q_interp_ = Eigen::Quaterniond(1,0,0,0); 
  Eigen::Matrix3d C_integral_interp_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d C_doubleintegral_interp_ = Eigen::Matrix3d::Zero(); 
  Eigen::Vector3d acc_integral_interp_ = Eigen::Vector3d::Zero(); 
  Eigen::Vector3d acc_doubleintegral_interp_ = Eigen::Vector3d::Zero(); 

  Eigen::Matrix3d cross_interp_ = Eigen::Matrix3d::Zero(); 

  Eigen::Matrix3d dalpha_db_g_interp_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d dv_db_g_interp_ = Eigen::Matrix3d::Zero(); 
  Eigen::Matrix3d dp_db_g_interp_ = Eigen::Matrix3d::Zero(); 

  Eigen::Vector3d omega_S_interp_ = Eigen::Vector3d::Zero(); 

};

struct State {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  kinematics::Transformation T_WS; 
  Eigen::Vector3d v_W; 
  Eigen::Vector3d b_g; 
  Eigen::Vector3d b_a; 
  Eigen::Vector3d omega_S; 
  Time timestamp; 
  StateId id; 
  ImuMeasurementDeque previousImuMeasurements; 
  bool isKeyframe; 
  AlignedMap<uint64_t, kinematics::Transformation> T_AiW; 
  std::set<StateId> covisibleFrameIds; 
  bool stateChanged; 
  bool isOnlineExtrinsics; 
  AlignedMap<uint64_t, kinematics::Transformation> extrinsics; 
  kinematics::Transformation T_GW; 
  AlignedVector<Eigen::Vector3d> gpsPoints; 
};

enum class TrackingQuality {
  Good, 
  Marginal, 
  Lost 
};

struct TrackingState {
  StateId id; 
  bool isKeyframe; 
  bool isLidarKeyframe; 
  TrackingQuality trackingQuality; 
  bool recognisedPlace; 
  bool isFullGraphOptimising; 
  StateId currentKeyframeId; 
};

class ConstantVelocityPropagator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ConstantVelocityPropagator(const State & stateStart, const State & stateEnd);

  Eigen::Quaterniond extrapolateAngle(const Time& timestamp);

  bool getState(const Time& timestamp, 
                kinematics::Transformation & T_WS_1,
                SpeedAndBias & sb_1, Eigen::Vector3d & omega_S);

  void newState(const State & newState);

private:

  State startState_;
  State endState_;
  Duration timeDelta_;
  Eigen::Vector3d r_s0_s1_;
};

class Trajectory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void update(const TrackingState & trackingState,
              std::shared_ptr<const AlignedMap<StateId, State> > updatedStates,
              std::set<StateId>& affectedStateIds);

  bool addImuMeasurement(const okvis::Time & stamp,
                         const Eigen::Vector3d & alpha,
                         const Eigen::Vector3d & omega,
                         State& state);

  bool getState(const Time & timestamp, State& state);

  bool getStates(const std::vector<Time> & timestamps,
                 std::vector<State>& states);

  bool getState(const StateId & stateId, State& state) const;

  const std::set<StateId>& stateIds() const {return allStateIds_;}

  const AlignedMap<StateId, State>& keyframeStates() const;

  bool clearCache(const StateId& keyframeStateId);

 private:
  AlignedMap<StateId, State> keyframeStates_; 
  std::map<StateId, StateId> keyframeIdByStateId_; 
  std::set<StateId> allStateIds_; 

  struct NonKeyframeState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    kinematics::Transformation T_Sk_S; 
    Eigen::Vector3d v_Sk; 
    Eigen::Vector3d omega_S_raw; 
    Time timestamp; 
    StateId keyframeId; 
    StateId id; 
    ImuMeasurementDeque previousImuMeasurements; 
  };

  std::map<StateId, AlignedMap<StateId, NonKeyframeState>> nonKeyframeStatesByKeyframeId_;

  std::map<uint64_t, StateId> statesByTimestampNs_;

  std::map<StateId, AlignedMap<uint64_t, Propagator>> propagatorsByKeyframeIdAndEndTimeNs_;

  std::unique_ptr<Propagator> livePropagator_;
  ImuMeasurementDeque imuMeasurements_; 
};

class ViInterface
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef std::function<void(const State &, const TrackingState &,
                             std::shared_ptr<const AlignedMap<StateId, State>>,
                             std::shared_ptr<const okvis::MapPointVector>)> OptimisedGraphCallback;

  ViInterface();

  virtual ~ViInterface();

  bool setImuCsvFile(std::fstream& csvFile);

  bool setImuCsvFile(const std::string& csvFileName);

  bool setTracksCsvFile(size_t cameraId, std::fstream& csvFile);

  bool setTracksCsvFile(size_t cameraId, const std::string& csvFileName);

  virtual bool addImages(const okvis::Time & stamp,
                         const std::map<size_t, cv::Mat> & images,
                         const std::map<size_t, cv::Mat> & depthImages
                         = std::map<size_t, cv::Mat>()) = 0;

  virtual bool addImuMeasurement(const okvis::Time & stamp,
                                 const Eigen::Vector3d & alpha,
                                 const Eigen::Vector3d & omega) = 0;

  virtual bool addDepthMeasurement(const okvis::Time &stamp,
                                   const cv::Mat &depthImage) = 0;

  virtual bool addLidarMeasurement(const okvis::Time & stamp,
                                   const Eigen::Vector3d & rayMeasurement) = 0;

  virtual void setOptimisedGraphCallback(const OptimisedGraphCallback & optimisedGraphCallback);

  virtual void setBlocking(bool blocking) = 0;

  virtual void display(std::map<std::string, cv::Mat> & images) = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:

  bool writeImuCsvDescription();

  bool writeTracksCsvDescription(size_t cameraId);

  OptimisedGraphCallback optimisedGraphCallback_; 

  std::shared_ptr<std::fstream> csvImuFile_;  
  typedef std::map<size_t, std::shared_ptr<std::fstream>> FilePtrMap; 
  FilePtrMap csvTracksFiles_;  

};

}  

#endif 
