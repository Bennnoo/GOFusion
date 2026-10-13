/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 * SPDX-License-Identifier: BSD-3-Clause
 */



#ifndef INCLUDE_OKVIS_DEPTHUTILS_HPP
#define INCLUDE_OKVIS_DEPTHUTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace okvis {

  inline cv::Size validatedDepthDownsampledSize(const int width,
                                                const int height,
                                                const size_t ratio,
                                                const char* context)
  {
    const std::string name = context ? context : "depth image";
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument(name + " has invalid dimensions "
                                  + std::to_string(width) + "x"
                                  + std::to_string(height));
    }
    if (ratio < 1) {
      throw std::invalid_argument(
          name + " depth_image_resolution_downsampling must be at least 1");
    }
    if (ratio > static_cast<size_t>(width)
        || ratio > static_cast<size_t>(height)) {
      throw std::invalid_argument(name + " downsampling ratio exceeds image dimensions");
    }
    return cv::Size(width / static_cast<int>(ratio),
                    height / static_cast<int>(ratio));
  }

  inline void downsampleDepth(const cv::Mat& depth,
                                      const size_t ratio,
                                      const cv::Size target_size,
                                      const float near_plane,
                                      const float far_plane,
                                      cv::Mat& depth_downsampled)
  {
    if (depth.empty() || depth.type() != CV_32FC1) {
      throw std::invalid_argument("Depth downsampling requires a non-empty CV_32FC1 image");
    }

    const cv::Size expected_size = validatedDepthDownsampledSize(
        depth.cols, depth.rows, ratio, "depth image");
    if (target_size != expected_size) {
      throw std::invalid_argument(
          "Downsampled depth target " + std::to_string(target_size.width) + "x"
          + std::to_string(target_size.height) + " does not match camera model "
          + std::to_string(expected_size.width) + "x"
          + std::to_string(expected_size.height));
    }

    depth_downsampled.create(target_size, CV_32FC1);
    const int block_size = static_cast<int>(ratio);
    if (ratio > std::numeric_limits<size_t>::max() / ratio) {
      throw std::overflow_error("Depth downsampling block size overflows size_t");
    }
    const size_t block_area = ratio * ratio;
    const bool limit_far = std::isfinite(far_plane) && far_plane > near_plane;
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    cv::parallel_for_(cv::Range(0, target_size.height), [&](const cv::Range& rows) {

      thread_local std::vector<float> samples;
      if (samples.size() < block_area) samples.resize(block_area);

      for (int y = rows.start; y < rows.end; ++y) {
        float* depth_out = depth_downsampled.ptr<float>(y);
        const int src_y0 = y * block_size;
        for (int x = 0; x < target_size.width; ++x) {
          const int src_x0 = x * block_size;
          size_t count = 0;
          for (int by = 0; by < block_size; ++by) {
            const float* depth_row = depth.ptr<float>(src_y0 + by);
            for (int bx = 0; bx < block_size; ++bx) {
              const int src_x = src_x0 + bx;
              const float value = depth_row[src_x];
              if (!std::isfinite(value) || !(value > 0.f)
                  || value < near_plane || (limit_far && value > far_plane)) {
                continue;
              }
              samples[count++] = value;
            }
          }

          if (count == 0) {
            depth_out[x] = invalid;
            continue;
          }

          std::sort(samples.begin(), samples.begin() + count,
                    [](const auto& lhs, const auto& rhs) {
                      return lhs < rhs;
                    });

          size_t foreground_count = count;
          for (size_t i = 1; i < count; ++i) {
            const float lower = samples[i - 1];
            const float gap_threshold = std::max(0.05f, 0.03f * lower);
            if (samples[i] - lower > gap_threshold) {
              foreground_count = i;
              break;
            }
          }

          const size_t median_index = (foreground_count - 1) / 2;
          depth_out[x] = samples[median_index];
        }
      }
    });
  }

} // namespace okvis
#endif
