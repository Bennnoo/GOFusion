/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */



#include <okvis/TrackingImageDownsampler.hpp>

#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace okvis {

TrackingImageDownsampler::TrackingImageDownsampler(
    const ViParameters& parameters)
    : factor_(parameters.frontend.tracking_image_resolution_downsampling) {
  if (factor_ == 0) {
    throw std::invalid_argument(
        "tracking image downsampling factor must be positive");
  }
  if (!enabled()) {
    return;
  }
  if (!parameters.effectiveTrackingCameraSystem) {
    throw std::invalid_argument(
        "tracking image downsampling is enabled without an effective camera "
        "system");
  }

  const auto& source_system = parameters.nCameraSystem;
  const auto& target_system = parameters.trackingCameraSystem();
  if (source_system.numCameras() != target_system.numCameras()) {
    throw std::invalid_argument(
        "source and effective tracking camera counts do not match");
  }

  source_sizes_.reserve(source_system.numCameras());
  target_sizes_.reserve(target_system.numCameras());
  for (size_t camera_index = 0;
       camera_index < source_system.numCameras(); ++camera_index) {
    const auto source = source_system.cameraGeometry(camera_index);
    const auto target = target_system.cameraGeometry(camera_index);
    source_sizes_.emplace_back(static_cast<int>(source->imageWidth()),
                               static_cast<int>(source->imageHeight()));
    target_sizes_.emplace_back(static_cast<int>(target->imageWidth()),
                               static_cast<int>(target->imageHeight()));
  }
}

cv::Mat TrackingImageDownsampler::process(
    const size_t camera_index, const cv::Mat& source) const {
  if (!enabled() || source.empty()) {
    return source;
  }
  if (camera_index >= source_sizes_.size()) {
    throw std::out_of_range("tracking image camera index is out of range");
  }
  if (source.size() != source_sizes_[camera_index]) {
    std::ostringstream message;
    message << "tracking camera " << camera_index << " expected source size "
            << source_sizes_[camera_index].width << "x"
            << source_sizes_[camera_index].height << " but received "
            << source.cols << "x" << source.rows;
    throw std::invalid_argument(message.str());
  }

  cv::Mat downsampled;
  cv::resize(source, downsampled, target_sizes_[camera_index], 0.0, 0.0,
             cv::INTER_AREA);
  return downsampled;
}

}  
