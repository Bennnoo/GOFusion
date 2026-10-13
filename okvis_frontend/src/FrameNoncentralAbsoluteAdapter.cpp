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
 * @file FrameNoncentralAbsoluteAdapter.cpp
 * @brief Source file for the FrameNoncentralAbsoluteAdapter class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/EucmCamera.hpp>

opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::FrameNoncentralAbsoluteAdapter(
    const Estimator &estimator,
    const okvis::cameras::NCameraSystem & nCameraSystem,
    std::shared_ptr<okvis::MultiFrame> frame) {

  size_t numCameras = nCameraSystem.numCameras();

  okvis::cameras::NCameraSystem::DistortionType distortionType= nCameraSystem.distortionType(0);
  for (size_t i = 1; i < nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception, distortionType == nCameraSystem.distortionType(i),
                            "mixed frame types are not supported yet")
  }

  for (size_t im = 0; im < numCameras; ++im) {

    camOffsets_.push_back(frame->T_SC(im)->r());
    camRotations_.push_back(frame->T_SC(im)->C());

    const size_t numK = frame->numKeypoints(im);
    double fu = 1.0;
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        fu = frame
            ->geometryAs<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> >(im)
            ->focalLengthU();
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        fu = frame
            ->geometryAs<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> >(im)
            ->focalLengthU();
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        fu =
            frame->geometryAs<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> >
            (im)->focalLengthU();
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        fu = frame->geometryAs<okvis::cameras::EucmCamera>(im)->focalLengthU();
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type")
        break;
    }
    for (size_t k = 0; k < numK; ++k) {
      uint64_t lmId = frame->landmarkId(im, k);

      if (lmId == 0 || !estimator.isLandmarkAdded(okvis::LandmarkId(lmId)))
        continue;
      okvis::MapPoint2 landmark;
      estimator.getLandmark(okvis::LandmarkId(lmId), landmark);
      if (landmark.observations.size() < 2)
        continue;

      const Eigen::Vector4d hp = landmark.point;

      if (fabs(hp[3]) < 1.0e-8)
        continue;

      points_.push_back(hp.head<3>() / hp[3]);

      Eigen::Vector3d bearing;
      Eigen::Vector2d keypoint;
      frame->getKeypoint(im, k, keypoint);
      double keypointStdDev;
      frame->getKeypointSize(im, k, keypointStdDev);
      keypointStdDev = 0.8 * keypointStdDev / 12.0;
      if(!frame->getBackProjection(im, k, bearing)) {
        bearing = Eigen::Vector3d(1,0,0);

      }

      sigmaAngles_.push_back(sqrt(2) * keypointStdDev * keypointStdDev / (fu * fu));

      bearing.normalize();
      bearingVectors_.push_back(bearing);

      camIndices_.push_back(im);

      keypointIndices_.push_back(k);

    }
  }
}

opengv::bearingVector_t opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getBearingVector(
    size_t index) const {
  assert(index < bearingVectors_.size());
  return bearingVectors_[index];
}

opengv::point_t opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getPoint(
    size_t index) const {
  assert(index < bearingVectors_.size());
  return points_[index];
}

opengv::translation_t opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getCamOffset(
    size_t index) const {
  return camOffsets_[camIndices_[index]];
}

opengv::rotation_t opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getCamRotation(
    size_t index) const {
  return camRotations_[camIndices_[index]];
}

size_t opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getNumberCorrespondences() const {
  return points_.size();
}

double opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::getSigmaAngle(
    size_t index) {
  return sigmaAngles_[index];
}
