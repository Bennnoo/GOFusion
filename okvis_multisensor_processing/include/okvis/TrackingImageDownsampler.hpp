/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */



#ifndef INCLUDE_OKVIS_TRACKINGIMAGEDOWNSAMPLER_HPP_
#define INCLUDE_OKVIS_TRACKINGIMAGEDOWNSAMPLER_HPP_

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include <okvis/Parameters.hpp>

namespace okvis {

class TrackingImageDownsampler {
 public:
  explicit TrackingImageDownsampler(const ViParameters& parameters);

  bool enabled() const noexcept { return factor_ > 1; }
  size_t factor() const noexcept { return factor_; }

  cv::Mat process(size_t camera_index, const cv::Mat& source) const;

 private:
  size_t factor_ = 1;
  std::vector<cv::Size> source_sizes_;
  std::vector<cv::Size> target_sizes_;
};

}  

#endif  
