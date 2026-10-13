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
 * @file implementation/MultiFrame.hpp
 * @brief Header implementation file for the MultiFrame class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <okvis/MultiFrame.hpp>

namespace okvis {

MultiFrame::MultiFrame() : id_(0) {}

MultiFrame::MultiFrame(const cameras::NCameraSystem & cameraSystem,
                       const okvis::Time & timestamp, uint64_t id)
    : timestamp_(timestamp),
      id_(id)
{
  resetCameraSystemAndFrames(cameraSystem);
}

MultiFrame::~MultiFrame()
{

}

void MultiFrame::resetCameraSystemAndFrames(
    const cameras::NCameraSystem & cameraSystem)
{
  cameraSystem_ = cameraSystem;
  frames_.clear();  
  frames_.resize(cameraSystem.numCameras());

  for(size_t c = 0; c<numFrames(); ++c){
    frames_[c].setGeometry(cameraSystem.cameraGeometry(c));
  }
}

const cameras::NCameraSystem & MultiFrame::cameraSystem() const {
  return cameraSystem_;
}

void MultiFrame::setTimestamp(const okvis::Time & timestamp)
{
  timestamp_ = timestamp;
}

void MultiFrame::setId(uint64_t id)
{
  id_ = id;
}

const okvis::Time & MultiFrame::timestamp() const
{
  return timestamp_;
}

uint64_t MultiFrame::id() const
{
  return id_;
}

size_t MultiFrame::numFrames() const
{
  return frames_.size();
}

std::shared_ptr<const okvis::kinematics::Transformation> MultiFrame::T_SC(size_t cameraIdx) const {
  return cameraSystem_.T_SC(cameraIdx);
}

void MultiFrame::setExtrinsics(size_t cameraIdx, const kinematics::Transformation &T_SCi)
{
  cameraSystem_.setExtrinsics(cameraIdx, T_SCi);
}

void MultiFrame::setImage(size_t cameraIdx, const cv::Mat & image)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setImage(image);
}

void MultiFrame::setDepthImage(size_t cameraIdx, const cv::Mat & depthImage)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setDepthImage(depthImage);
}

void MultiFrame::setGeometry(
    size_t cameraIdx, std::shared_ptr<const cameras::CameraBase> cameraGeometry)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setGeometry(cameraGeometry);
}

void MultiFrame::setDetector(size_t cameraIdx,
                             std::shared_ptr<cv::FeatureDetector> detector)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setDetector(detector);
}

void MultiFrame::setExtractor(
    size_t cameraIdx, std::shared_ptr<cv::DescriptorExtractor> extractor)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  frames_[cameraIdx].setExtractor(extractor);
}

const cv::Mat & MultiFrame::image(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].image();
}

const cv::Mat & MultiFrame::depthImage(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].depthImage();
}

std::shared_ptr<const cameras::CameraBase> MultiFrame::geometry(
    size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].geometry();
}

template<class GEOMETRY_T>
std::shared_ptr<const GEOMETRY_T> MultiFrame::geometryAs(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].geometryAs<GEOMETRY_T>();
}

int MultiFrame::detect(size_t cameraIdx)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].detect();
}

int MultiFrame::describe(size_t cameraIdx)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].describe();
}

int MultiFrame::computeBackProjections(size_t cameraIdx) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].computeBackProjections();
}

bool MultiFrame::getBackProjection(size_t cameraIdx, size_t keypointIdx,
                                   Eigen::Vector3d& backProjection) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getBackProjection(keypointIdx, backProjection);
}

bool MultiFrame::getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                               cv::KeyPoint & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getCvKeypoint(keypointIdx, keypoint);
}

bool MultiFrame::getKeypoint(size_t cameraIdx, size_t keypointIdx,
                             Eigen::Vector2d & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getKeypoint(keypointIdx, keypoint);
}

bool MultiFrame::getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                                 double & keypointSize) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getKeypointSize(keypointIdx, keypointSize);
}

const unsigned char * MultiFrame::keypointDescriptor(size_t cameraIdx,
                                                     size_t keypointIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].keypointDescriptor(keypointIdx);
}

bool MultiFrame::setLandmarkId(size_t cameraIdx, size_t keypointIdx,
                               uint64_t landmarkId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].setLandmarkId(keypointIdx, landmarkId);
}

bool MultiFrame::setLandmark(size_t cameraIdx, size_t keypointIdx,
                             const Eigen::Vector4d & landmark, bool isInitialised) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].setLandmark(keypointIdx, landmark, isInitialised);
}

uint64_t MultiFrame::landmarkId(size_t cameraIdx, size_t keypointIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].landmarkId(keypointIdx);
}

bool MultiFrame::getLandmark(size_t cameraIdx, size_t keypointIdx, Eigen::Vector4d & landmark,
                             bool & isInitialised) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getLandmark(keypointIdx, landmark, isInitialised);
}

size_t MultiFrame::numKeypoints(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].numKeypoints();
}

bool MultiFrame::resetKeypoints(size_t cameraIdx, const std::vector<cv::KeyPoint> & keypoints){
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].resetKeypoints(keypoints);
}

bool MultiFrame::resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].resetDescriptors(descriptors);
}

size_t MultiFrame::numKeypoints() const
{
  size_t numKeypoints = 0;
  for (size_t i = 0; i < frames_.size(); ++i) {
    numKeypoints += frames_[i].numKeypoints();
  }
  return numKeypoints;
}

}
