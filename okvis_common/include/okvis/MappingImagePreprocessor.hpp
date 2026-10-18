/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */



#ifndef INCLUDE_OKVIS_MAPPINGIMAGEPREPROCESSOR_HPP_
#define INCLUDE_OKVIS_MAPPINGIMAGEPREPROCESSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>

#include <okvis/Measurements.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

class MappingImagePreprocessor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class Geometry {
    Target,
    Distorted,
    Pinhole,
    SameAsRgb,
    SameAsDepth,
  };

  struct ModalityConfig {
    std::string stream;
    Geometry geometry = Geometry::Target;
    std::shared_ptr<const cameras::CameraBase> sourceGeometry;
  };

  struct Config {
    bool enabled = false;
    size_t targetCameraIndex = std::numeric_limits<size_t>::max();
    std::shared_ptr<const cameras::CameraBase> targetGeometry;
    std::shared_ptr<const kinematics::Transformation> targetT_SC;
    ModalityConfig rgb;
    ModalityConfig depth;
  };

  explicit MappingImagePreprocessor(Config config);

  bool enabled() const { return config_.enabled; }
  size_t targetCameraIndex() const { return config_.targetCameraIndex; }
  const std::string& rgbStream() const { return config_.rgb.stream; }
  const std::string& depthStream() const { return config_.depth.stream; }
  const std::shared_ptr<const kinematics::Transformation>& targetT_SC() const {
    return config_.targetT_SC;
  }

  std::shared_ptr<const cameras::CameraBase> targetGeometry(
      size_t downsampling = 1) const;

  uint64_t geometryRevision(size_t downsampling = 1) const;

  bool process(size_t cameraIndex,
               CameraMeasurement& measurement,
               size_t downsampling = 1) const;

  static Geometry geometryFromString(const std::string& value);
  static const char* geometryName(Geometry geometry);

 private:
  struct Remap {
    cv::Size sourceSize;
    cv::Mat map1;
    cv::Mat map2;
    bool integerCoordinates = false;
    bool identity = true;
  };

  struct FactorCache {
    std::shared_ptr<const cameras::CameraBase> targetGeometry;
    std::shared_ptr<const Remap> rgb;
    std::shared_ptr<const Remap> depth;
  };

  std::shared_ptr<const FactorCache> factorCache(size_t downsampling) const;
  std::shared_ptr<const Remap> makeRemap(
      const ModalityConfig& modality,
      const std::shared_ptr<const cameras::CameraBase>& target) const;
  static void processMat(const Remap& remap,
                         cv::Mat& image,
                         int interpolation,
                         const char* modality);
  static bool samePinholeGeometry(const cameras::CameraBase& lhs,
                                  const cameras::CameraBase& rhs);

  Config config_;
  mutable std::mutex cacheMutex_;
  mutable std::map<size_t, std::shared_ptr<const FactorCache>> factorCaches_;
};

}  

#endif  
