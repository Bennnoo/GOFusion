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
 * @file FrameTypedefs.hpp
 * @brief This file contains useful typedefs and structs related to frames.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_FRAMETYPEDEFS_HPP_
#define INCLUDE_OKVIS_FRAMETYPEDEFS_HPP_

#include <map>
#include <unordered_map>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

template <typename Parameter>
class NamedId {
public:

  NamedId() = default;

  explicit NamedId(uint64_t const& value) : value_(value) {}

  explicit NamedId(uint64_t&& value) : value_(std::move(value)) {}

  uint64_t& value() { return value_; }

  uint64_t const& value() const {return value_; }

  bool operator==(const NamedId<Parameter>& other) const {return value_ == other.value_; }

  bool operator!=(const NamedId<Parameter>& other) const {return value_ != other.value_; }

  bool operator>(const NamedId<Parameter>& other) const {return value_ > other.value_; }

  bool operator<(const NamedId<Parameter>& other) const {return value_ < other.value_; }

  NamedId operator+(uint64_t add) const {return NamedId(value_ + add);}

  bool isInitialised() const {return value_ != 0; }

private:
  uint64_t value_ = 0; 
};

using LandmarkId = NamedId<struct LandmarkIdParameter>;

using StateId = NamedId<struct StateIdParameter>;

struct KeypointIdentifier
{

  KeypointIdentifier(uint64_t fi = 0, size_t ci = 0, size_t ki = 0)
      : frameId(fi),
        cameraIndex(ci),
        keypointIndex(ki)
  {
  }

  uint64_t frameId;     
  size_t cameraIndex;   
  size_t keypointIndex; 

  uint64_t getFrameId() const
  {
    return frameId;
  }

  void setFrameId(uint64_t fid)
  {
    frameId = fid;
  }

  bool isBinaryEqual(const KeypointIdentifier & rhs) const
  {
    return frameId == rhs.frameId && cameraIndex == rhs.cameraIndex
        && keypointIndex == rhs.keypointIndex;
  }

  bool operator==(const KeypointIdentifier & rhs) const
  {
    return isBinaryEqual(rhs);
  }

  bool operator<(const KeypointIdentifier & rhs) const
  {

    if (frameId == rhs.frameId) {
      if (cameraIndex == rhs.cameraIndex) {
        return keypointIndex < rhs.keypointIndex;
      } else {
        return cameraIndex < rhs.cameraIndex;
      }
    }
    return frameId < rhs.frameId;
  }

};

struct Match
{

  Match(size_t idxA_, size_t idxB_, float distance_)
      : idxA(idxA_),
        idxB(idxB_),
        distance(distance_)
  {
  }
  size_t idxA;    
  size_t idxB;    
  float distance; 
};

typedef std::vector<Match> Matches;

struct MapPoint
{

  MapPoint()
      : id(0),
        point(Eigen::Matrix<double,4,1,Eigen::DontAlign>(0.0, 0.0, 0.0, 1.0)),
        quality(0.0),
        stateId(0)
  {
  }

  MapPoint(uint64_t id, const Eigen::Matrix<double,4,1,Eigen::DontAlign> & point,
           double quality, uint64_t stateId = 0)
      : id(id),
        point(point),
        quality(quality),
        stateId(stateId)
  {
  }
  uint64_t id;            
  Eigen::Matrix<double,4,1,Eigen::DontAlign> point;  
  double quality;         
  uint64_t stateId;       
};

struct MapPoint2
{
  LandmarkId id;            
  Eigen::Matrix<double,4,1,Eigen::DontAlign> point;  
  std::set<KeypointIdentifier> observations; 
  bool isInitialised;      
  double quality;     
};


template<typename KEY_T, typename VALUE_T>
using AlignedMap = std::map<KEY_T, VALUE_T, std::less<KEY_T>,
Eigen::aligned_allocator<std::pair<const KEY_T, VALUE_T>>>;

template<typename KEY_T, typename VALUE_T>
using AlignedUnorderedMap = std::unordered_map<KEY_T, VALUE_T, std::hash<KEY_T>, std::equal_to<KEY_T>,  
Eigen::aligned_allocator<std::pair<const KEY_T, VALUE_T>>>;

template<typename VALUE_T>
using AlignedVector = std::vector<VALUE_T, Eigen::aligned_allocator<VALUE_T>>;

template<typename VALUE_T>
using AlignedSet = std::set<VALUE_T, Eigen::aligned_allocator<VALUE_T>>;

typedef std::vector<MapPoint> MapPointVector;


typedef std::map<LandmarkId, MapPoint2 > MapPoints;

typedef AlignedMap<uint64_t, okvis::kinematics::Transformation> TransformationMap;

struct Observation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Observation(size_t keypointIdx,
              const Eigen::Vector2d& keypointMeasurement,
              double keypointSize,
              size_t cameraIdx,
              uint64_t frameId,
              const Eigen::Vector4d& landmark_W,
              uint64_t landmarkId, bool isInitialized)
      : keypointIdx(keypointIdx),
        cameraIdx(cameraIdx),
        frameId(frameId),
        keypointMeasurement(keypointMeasurement),
        keypointSize(keypointSize),
        landmark_W(landmark_W),
        landmarkId(landmarkId),
        isInitialized(isInitialized)
  {
  }
  Observation()
      : keypointIdx(0),
        cameraIdx(std::numeric_limits<size_t>::max()),
        frameId(0),
        keypointSize(0),
        landmarkId(0),
        isInitialized(false)
  {
  }
  size_t keypointIdx; 
  size_t cameraIdx;  
  uint64_t frameId;  
  Eigen::Vector2d keypointMeasurement;  
  double keypointSize;  
  Eigen::Vector4d landmark_W;  
  uint64_t landmarkId;  
  bool isInitialized;  
};

typedef std::vector<Observation, Eigen::aligned_allocator<Observation> > ObservationVector;

typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;

}  

#endif 
