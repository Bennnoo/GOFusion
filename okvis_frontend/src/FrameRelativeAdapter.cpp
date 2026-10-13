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
 * @file FrameRelativeAdapter.cpp
 * @brief Source file for the FrameRelativeAdapter class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <opengv/relative_pose/FrameRelativeAdapter.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/EucmCamera.hpp>

opengv::relative_pose::FrameRelativeAdapter::FrameRelativeAdapter(const Estimator &estimator,
    const okvis::cameras::NCameraSystem & nCameraSystem, uint64_t multiFrameIdA,
    size_t camIdA, uint64_t multiFrameIdB, size_t camIdB) {

  std::shared_ptr<okvis::MultiFrame> frameAPtr =
      estimator.multiFrame(okvis::StateId(multiFrameIdA));
  std::shared_ptr<okvis::MultiFrame> frameBPtr =
      estimator.multiFrame(okvis::StateId(multiFrameIdB));

  okvis::cameras::NCameraSystem::DistortionType distortionTypeA =
      nCameraSystem.distortionType(camIdA);
  okvis::cameras::NCameraSystem::DistortionType distortionTypeB =
      nCameraSystem.distortionType(camIdB);

  double fu1 = 0;
  size_t numKeypointsA = frameAPtr->numKeypoints(camIdA);
  switch (distortionTypeA) {
    case okvis::cameras::NCameraSystem::RadialTangential: {
      fu1 = frameAPtr
          ->geometryAs<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion> >(camIdA)
          ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      fu1 =
          frameAPtr
              ->geometryAs<
                  okvis::cameras::PinholeCamera<
                      okvis::cameras::EquidistantDistortion> >(camIdA)
              ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      fu1 = frameAPtr
          ->geometryAs<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> >(camIdA)
          ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::NoDistortion: {
      fu1 = frameAPtr->geometryAs<okvis::cameras::EucmCamera>(camIdA)->focalLengthU();
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type")
      break;
  }
  double fu2 = 0.0;
  size_t numKeypointsB = frameBPtr->numKeypoints(camIdB);
  switch (distortionTypeB) {
    case okvis::cameras::NCameraSystem::RadialTangential: {
      fu2 = frameAPtr
          ->geometryAs<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion> >(camIdB)
          ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      fu2 =
          frameAPtr
              ->geometryAs<
                  okvis::cameras::PinholeCamera<
                      okvis::cameras::EquidistantDistortion> >(camIdB)
              ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      fu2 = frameAPtr
          ->geometryAs<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> >(camIdB)
          ->focalLengthU();
      break;
    }
    case okvis::cameras::NCameraSystem::NoDistortion: {
      fu2 = frameAPtr->geometryAs<okvis::cameras::EucmCamera>(camIdB)->focalLengthU();
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type")
      break;
  }

  bearingVectors1_.resize(numKeypointsA);
  bearingVectors2_.resize(numKeypointsB);
  sigmaAngles1_.resize(numKeypointsA);
  sigmaAngles2_.resize(numKeypointsB);

  matches_.reserve(std::min(numKeypointsA, numKeypointsB));
  std::map<uint64_t, size_t> idMap;
  for (size_t k = 0; k < numKeypointsB; ++k) {

    uint64_t lmId = frameBPtr->landmarkId(camIdB, k);
    if (lmId == 0)
      continue;

    if (!estimator.isLandmarkAdded(okvis::LandmarkId(lmId)))
      continue;

    idMap.insert(std::pair<uint64_t, size_t>(lmId, k));
  }

  for (size_t k = 0; k < numKeypointsA; ++k) {

    uint64_t lmId = frameAPtr->landmarkId(camIdA, k);
    if (lmId == 0)
      continue;

    std::map<uint64_t, size_t>::const_iterator it = idMap.find(lmId);
    if (it != idMap.end()) {

      matches_.push_back(okvis::Match(k, it->second, 0.0));
    }
  }

  for (size_t k = 0; k < matches_.size(); ++k) {
    const size_t idx1 = matches_[k].idxA;
    const size_t idx2 = matches_[k].idxB;
    Eigen::Vector2d keypoint;
    double keypointStdDev;
    frameAPtr->getKeypoint(camIdA, idx1, keypoint);
    frameAPtr->getKeypointSize(camIdA, idx1, keypointStdDev);
    keypointStdDev = 0.8 * keypointStdDev / 12.0;
    sigmaAngles1_[idx1] = sqrt(2) * keypointStdDev * keypointStdDev
        / (fu1 * fu1);
    if(!frameAPtr->getBackProjection(camIdA, idx1, bearingVectors1_[idx1])) {
      bearingVectors1_[idx1] = Eigen::Vector3d(1,0,0);

    }
    bearingVectors1_[idx1].normalize();

    frameBPtr->getKeypoint(camIdB, idx2, keypoint);
    frameBPtr->getKeypointSize(camIdB, idx2, keypointStdDev);
    keypointStdDev = 0.8 * keypointStdDev / 12.0;
    sigmaAngles2_[idx2] = sqrt(2) * keypointStdDev * keypointStdDev
        / (fu2 * fu2);
    if(!frameBPtr->getBackProjection(camIdB, idx2, bearingVectors2_[idx2])) {
      bearingVectors1_[idx1] = Eigen::Vector3d(1,0,0);

    }
    bearingVectors2_[idx2].normalize();
  }
}

opengv::bearingVector_t opengv::relative_pose::FrameRelativeAdapter::getBearingVector1(
    size_t index) const {
  return bearingVectors1_[matches_[index].idxA];
}

opengv::bearingVector_t opengv::relative_pose::FrameRelativeAdapter::getBearingVector2(
    size_t index) const {
  return bearingVectors2_[matches_[index].idxB];
}

opengv::translation_t opengv::relative_pose::FrameRelativeAdapter::getCamOffset1(
    size_t ) const {

  return Eigen::Vector3d::Zero();
}

opengv::rotation_t opengv::relative_pose::FrameRelativeAdapter::getCamRotation1(
    size_t ) const {

  return Eigen::Matrix3d::Identity();
}

opengv::translation_t opengv::relative_pose::FrameRelativeAdapter::getCamOffset2(
    size_t ) const {

  return Eigen::Vector3d::Zero();
}

opengv::rotation_t opengv::relative_pose::FrameRelativeAdapter::getCamRotation2(
    size_t ) const {

  return Eigen::Matrix3d::Identity();
}

size_t opengv::relative_pose::FrameRelativeAdapter::getNumberCorrespondences() const {
  return matches_.size();
}

double opengv::relative_pose::FrameRelativeAdapter::getSigmaAngle1(
    size_t index) {
  return sigmaAngles1_[matches_[index].idxA];
}

double opengv::relative_pose::FrameRelativeAdapter::getSigmaAngle2(
    size_t index) {
  return sigmaAngles2_[matches_[index].idxB];
}
