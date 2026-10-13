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
 * @file implementation/Frame.hpp
 * @brief Header implementation file for the Frame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#pragma once

#include <okvis/Frame.hpp>

namespace okvis {

Frame::Frame(const cv::Mat & image,
             std::shared_ptr<cameras::CameraBase> & cameraGeometry,
             std::shared_ptr<cv::FeatureDetector> & detector,
             std::shared_ptr<cv::DescriptorExtractor> & extractor)
    : image_(image),
      cameraGeometry_(cameraGeometry),
      detector_(detector),
      extractor_(extractor)
{
}

void Frame::setImage(const cv::Mat & image)
{

  image_ = image;
}

void Frame::setDepthImage(const cv::Mat & depthImage)
{
  depthImage_ = depthImage;
}

void Frame::setGeometry(std::shared_ptr<const cameras::CameraBase> cameraGeometry)
{
  cameraGeometry_ = cameraGeometry;
}

void Frame::setDetector(std::shared_ptr<cv::FeatureDetector> detector)
{
  detector_ = detector;
}

void Frame::setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor)
{
  extractor_ = extractor;
}

const cv::Mat & Frame::image() const
{
  return image_;
}

const cv::Mat & Frame::depthImage() const
{
  return depthImage_;
}

std::shared_ptr<const cameras::CameraBase> Frame::geometry() const
{
  return cameraGeometry_;
}

template<class GEOMETRY_T>
std::shared_ptr<const GEOMETRY_T> Frame::geometryAs() const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception, std::dynamic_pointer_cast<const GEOMETRY_T>(cameraGeometry_),
      "incorrect pointer cast requested. " << cameraGeometry_->distortionType())
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#else
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#endif
}

int Frame::detect()
{

  keypoints_.clear();
  descriptors_.resize(0);

  landmarkIds_.clear();

  OKVIS_ASSERT_TRUE_DBG(Exception, detector_ != nullptr,
                        "Detector not initialised!")
  detector_->detect(image_, keypoints_);
  return int(keypoints_.size());
}

int Frame::describe()
{

  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != nullptr,
                        "Detector not initialised!")

  extractor_->compute(image_, keypoints_, descriptors_);

  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(),0);
  landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
        keypoints_.size(), Eigen::Vector4d(0,0,0,0));
  landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);
  return int(keypoints_.size());
}

int Frame::computeBackProjections() {
  backProjections_.resize(keypoints_.size());
  backProjectionsValid_.resize(keypoints_.size());
  int ctr=0;
  for (size_t k = 0; k < keypoints_.size(); ++k) {
    cv::KeyPoint& ckp = keypoints_[k];

    const bool success = cameraGeometry_->backProject(
          Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &backProjections_[k]);
    backProjectionsValid_[k] = success;
    if(success) {
      ctr++;
    }
  }
  return ctr;
}

bool Frame::getBackProjection(size_t keypointIdx, Eigen::Vector3d& backProjection) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  backProjection = backProjections_[keypointIdx];
  return backProjectionsValid_[keypointIdx];
#else
  backProjection = backProjections_[keypointIdx];
  return backProjectionsValid_[keypointIdx];
#endif
}

bool Frame::getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypoint = keypoints_[keypointIdx];
  return keypointIdx < keypoints_.size();
#else
  keypoint = keypoints_[keypointIdx];
  return true;
#endif
}

bool Frame::getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x,
                             keypoints_[keypointIdx].pt.y);
  return keypointIdx < keypoints_.size();
#else
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
  return true;
#endif
}

bool Frame::getKeypointSize(size_t keypointIdx, double & keypointSize) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypointSize = double(keypoints_[keypointIdx].size);
  return keypointIdx < keypoints_.size();
#else
  keypointSize = double(keypoints_[keypointIdx].size);
  return true;
#endif
}

const unsigned char * Frame::keypointDescriptor(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "<< keypoints_.size())
  return descriptors_.data + size_t(descriptors_.cols) * keypointIdx;
#else
  return descriptors_.data + size_t(descriptors_.cols) * keypointIdx;
#endif
}

bool Frame::setLandmarkId(size_t keypointIdx, uint64_t landmarkId)
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarkIds_.size(),
      "keypointIdx " << keypointIdx
        << "out of range: landmarkIds_ has size "<< landmarkIds_.size())
  landmarkIds_[keypointIdx] = landmarkId;
  return keypointIdx < keypoints_.size();
#else
  landmarkIds_[keypointIdx] = landmarkId;
  return true;
#endif
}

bool Frame::setLandmark(size_t keypointIdx, const Eigen::Vector4d & landmark, bool isInitialised)
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarks_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarks_ has size " << landmarks_.size())
  landmarks_[keypointIdx] = landmark;
  landmarkInitialisations_[keypointIdx] = isInitialised;
  return keypointIdx < landmarks_.size();
#else
  landmarks_[keypointIdx] = landmark;
  landmarkInitialisations_[keypointIdx] = isInitialised;
  return true;
#endif
}

uint64_t Frame::landmarkId(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarkIds_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarkIds has size " << landmarkIds_.size())
  return landmarkIds_[keypointIdx];
#else
  return landmarkIds_[keypointIdx];
#endif
}

bool Frame::getLandmark(size_t keypointIdx, Eigen::Vector4d & landmark, bool & isInitialised) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarks_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarks has size " << landmarks_.size())
  landmark = landmarks_[keypointIdx];
  isInitialised = landmarkInitialisations_[keypointIdx];
  return keypointIdx < landmarks_.size();
#else
  landmark = landmarks_[keypointIdx];
  isInitialised = landmarkInitialisations_[keypointIdx];
  return true;
#endif
}

inline bool Frame::resetKeypoints(const std::vector<cv::KeyPoint> & keypoints) {
  keypoints_ = keypoints;

  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(),0);
  landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
    keypoints_.size(), Eigen::Vector4d(0,0,0,0));
  landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);

  return true;
}

inline bool Frame::resetDescriptors(const cv::Mat & descriptors) {
  descriptors_ = descriptors;
  return true;
}

size_t Frame::numKeypoints() const {
  return keypoints_.size();
}

}  
