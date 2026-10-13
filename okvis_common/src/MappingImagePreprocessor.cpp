/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */



#include <okvis/MappingImagePreprocessor.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

namespace okvis {

namespace {

using TargetCamera = cameras::PinholeCamera<cameras::NoDistortion>;

std::string lowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

Eigen::Vector4d pinholeIntrinsics(const cameras::CameraBase& camera,
                                  const char* description) {
  Eigen::VectorXd intrinsics;
  camera.getIntrinsics(intrinsics);
  if (intrinsics.size() < 4 || !intrinsics.head<4>().allFinite()
      || intrinsics[0] == 0.0 || intrinsics[1] == 0.0) {
    throw std::invalid_argument(std::string(description)
                                + " has invalid pinhole intrinsics");
  }
  return intrinsics.head<4>();
}

}  

MappingImagePreprocessor::MappingImagePreprocessor(Config config)
    : config_(std::move(config)) {
  if (config_.rgb.stream.empty()) {
    config_.rgb.stream = "rgb0";
  }
  if (config_.depth.stream.empty()) {
    config_.depth.stream = "depth0";
  }

  if (!config_.enabled) {
    return;
  }
  if (!config_.targetGeometry || !config_.targetT_SC
      || config_.targetCameraIndex == std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(
        "mapping_input_parameters requires a valid target camera");
  }
  (void) pinholeIntrinsics(*config_.targetGeometry, "mapping target camera");

  const auto validateSource = [](const ModalityConfig& modality,
                                 const char* name) {
    if ((modality.geometry == Geometry::Distorted
         || modality.geometry == Geometry::Pinhole)
        && !modality.sourceGeometry) {
      throw std::invalid_argument(std::string("mapping ") + name
                                  + " source geometry is missing");
    }
  };
  validateSource(config_.rgb, "RGB");
  validateSource(config_.depth, "depth");

  if (config_.rgb.geometry == Geometry::SameAsRgb) {
    throw std::invalid_argument("mapping RGB geometry cannot alias itself");
  }
  if (config_.depth.geometry == Geometry::SameAsDepth) {
    throw std::invalid_argument("mapping depth geometry cannot alias itself");
  }
  if (config_.rgb.geometry == Geometry::SameAsDepth
      && config_.depth.geometry == Geometry::SameAsRgb) {
    throw std::invalid_argument(
        "mapping RGB/depth geometry aliases form a cycle");
  }
}

MappingImagePreprocessor::Geometry
MappingImagePreprocessor::geometryFromString(const std::string& input) {
  const std::string value = lowerCopy(input);
  if (value == "target") return Geometry::Target;
  if (value == "distorted") return Geometry::Distorted;
  if (value == "pinhole") return Geometry::Pinhole;
  if (value == "same_as_rgb") return Geometry::SameAsRgb;
  if (value == "same_as_depth") return Geometry::SameAsDepth;
  throw std::invalid_argument("unknown mapping input geometry: " + input);
}

const char* MappingImagePreprocessor::geometryName(const Geometry geometry) {
  switch (geometry) {
    case Geometry::Target: return "target";
    case Geometry::Distorted: return "distorted";
    case Geometry::Pinhole: return "pinhole";
    case Geometry::SameAsRgb: return "same_as_rgb";
    case Geometry::SameAsDepth: return "same_as_depth";
  }
  return "unknown";
}

uint64_t MappingImagePreprocessor::geometryRevision(
    const size_t downsampling) const {
  if (!config_.enabled) return 0;
  if (downsampling < 1) {
    throw std::invalid_argument("mapping image downsampling must be at least 1");
  }
  constexpr uint64_t kMappingGeometryTag = 0x4f4b4d5000000000ull;
  return kMappingGeometryTag
         ^ (static_cast<uint64_t>(config_.targetCameraIndex & 0xffffu) << 32)
         ^ static_cast<uint64_t>(downsampling);
}

bool MappingImagePreprocessor::samePinholeGeometry(
    const cameras::CameraBase& lhs,
    const cameras::CameraBase& rhs) {
  if (lhs.imageWidth() != rhs.imageWidth()
      || lhs.imageHeight() != rhs.imageHeight()) {
    return false;
  }
  const Eigen::Vector4d lhs_k = pinholeIntrinsics(lhs, "source camera");
  const Eigen::Vector4d rhs_k = pinholeIntrinsics(rhs, "target camera");
  return (lhs_k - rhs_k).cwiseAbs().maxCoeff() <= 1.0e-9;
}

std::shared_ptr<const MappingImagePreprocessor::Remap>
MappingImagePreprocessor::makeRemap(
    const ModalityConfig& modality,
    const std::shared_ptr<const cameras::CameraBase>& target) const {
  auto remap = std::make_shared<Remap>();
  std::shared_ptr<const cameras::CameraBase> source;
  if (modality.geometry == Geometry::Target) {
    source = config_.targetGeometry;
  } else {
    source = modality.sourceGeometry;
  }
  if (!source || !target) {
    throw std::invalid_argument("mapping remap camera geometry is missing");
  }
  remap->sourceSize = cv::Size(static_cast<int>(source->imageWidth()),
                               static_cast<int>(source->imageHeight()));

  if ((modality.geometry == Geometry::Target
       || modality.geometry == Geometry::Pinhole)
      && samePinholeGeometry(*source, *target)) {
    remap->identity = true;
    return remap;
  }

  remap->identity = false;
  const Eigen::Vector4d target_k = pinholeIntrinsics(*target, "target camera");
  cv::Mat map_x(static_cast<int>(target->imageHeight()),
                static_cast<int>(target->imageWidth()), CV_32FC1);
  cv::Mat map_y(map_x.size(), CV_32FC1);

  cv::parallel_for_(cv::Range(0, map_x.rows), [&](const cv::Range& rows) {
    for (int y = rows.start; y < rows.end; ++y) {
      float* x_ptr = map_x.ptr<float>(y);
      float* y_ptr = map_y.ptr<float>(y);
      for (int x = 0; x < map_x.cols; ++x) {
        const Eigen::Vector3d ray(
            (static_cast<double>(x) - target_k[2]) / target_k[0],
            (static_cast<double>(y) - target_k[3]) / target_k[1],
            1.0);
        Eigen::Vector2d source_pixel;
        const cameras::ProjectionStatus status = source->project(ray, &source_pixel);
        if ((status != cameras::ProjectionStatus::Successful
             && status != cameras::ProjectionStatus::OutsideImage)
            || !source_pixel.allFinite()) {
          x_ptr[x] = -1.0f;
          y_ptr[x] = -1.0f;
        } else {
          x_ptr[x] = static_cast<float>(source_pixel.x());
          y_ptr[x] = static_cast<float>(source_pixel.y());
        }
      }
    }
  });

  cv::convertMaps(map_x, map_y, remap->map1, remap->map2, CV_16SC2, false);
  remap->integerCoordinates = cv::countNonZero(remap->map2) == 0;
  return remap;
}

std::shared_ptr<const MappingImagePreprocessor::FactorCache>
MappingImagePreprocessor::factorCache(const size_t downsampling) const {
  if (downsampling < 1) {
    throw std::invalid_argument("mapping image downsampling must be at least 1");
  }

  std::lock_guard<std::mutex> lock(cacheMutex_);
  const auto existing = factorCaches_.find(downsampling);
  if (existing != factorCaches_.end()) return existing->second;

  if (!config_.targetGeometry) {
    throw std::invalid_argument("mapping target geometry is missing");
  }
  const int source_width = static_cast<int>(config_.targetGeometry->imageWidth());
  const int source_height = static_cast<int>(config_.targetGeometry->imageHeight());
  if (downsampling > static_cast<size_t>(source_width)
      || downsampling > static_cast<size_t>(source_height)) {
    throw std::invalid_argument("mapping image downsampling exceeds target dimensions");
  }
  const int width = source_width / static_cast<int>(downsampling);
  const int height = source_height / static_cast<int>(downsampling);
  const double sx = static_cast<double>(width) / source_width;
  const double sy = static_cast<double>(height) / source_height;
  const Eigen::Vector4d source_k =
      pinholeIntrinsics(*config_.targetGeometry, "mapping target camera");

  auto cache = std::make_shared<FactorCache>();
  cache->targetGeometry = std::make_shared<TargetCamera>(
      width, height,
      sx * source_k[0], sy * source_k[1],
      (source_k[2] + 0.5) * sx - 0.5,
      (source_k[3] + 0.5) * sy - 0.5,
      cameras::NoDistortion());

  auto resolve = [&](const ModalityConfig& modality,
                     const std::shared_ptr<const Remap>& rgb,
                     const std::shared_ptr<const Remap>& depth) {
    if (modality.geometry == Geometry::SameAsRgb) {
      if (!rgb) throw std::invalid_argument("mapping RGB remap alias is unresolved");
      return rgb;
    }
    if (modality.geometry == Geometry::SameAsDepth) {
      if (!depth) throw std::invalid_argument("mapping depth remap alias is unresolved");
      return depth;
    }
    return makeRemap(modality, cache->targetGeometry);
  };

  if (config_.rgb.geometry == Geometry::SameAsDepth) {
    cache->depth = resolve(config_.depth, nullptr, nullptr);
    cache->rgb = cache->depth;
  } else {
    cache->rgb = resolve(config_.rgb, nullptr, nullptr);
    cache->depth = (config_.depth.geometry == Geometry::SameAsRgb)
                       ? cache->rgb
                       : resolve(config_.depth, cache->rgb, nullptr);
  }

  LOG(INFO) << "Mapping input geometry ready: factor=" << downsampling
            << " target=" << width << "x" << height
            << " RGB=" << geometryName(config_.rgb.geometry)
            << " (source " << cache->rgb->sourceSize.width << "x"
            << cache->rgb->sourceSize.height << ", "
            << (cache->rgb->identity ? "pass-through" : "fixed-point remap")
            << ") depth=" << geometryName(config_.depth.geometry)
            << " (source " << cache->depth->sourceSize.width << "x"
            << cache->depth->sourceSize.height << ", "
            << (cache->depth->identity ? "pass-through" : "fixed-point remap")
            << ")";

  const auto inserted = factorCaches_.emplace(downsampling, cache);
  return inserted.first->second;
}

std::shared_ptr<const cameras::CameraBase>
MappingImagePreprocessor::targetGeometry(const size_t downsampling) const {
  if (!config_.enabled) return config_.targetGeometry;
  return factorCache(downsampling)->targetGeometry;
}

void MappingImagePreprocessor::processMat(const Remap& remap,
                                          cv::Mat& image,
                                          const int interpolation,
                                          const char* modality) {
  if (image.empty()) return;
  if (image.size() != remap.sourceSize) {
    throw std::invalid_argument(
        std::string("mapping ") + modality + " image has size "
        + std::to_string(image.cols) + "x" + std::to_string(image.rows)
        + ", expected " + std::to_string(remap.sourceSize.width) + "x"
        + std::to_string(remap.sourceSize.height));
  }
  if ((interpolation == cv::INTER_NEAREST) && image.channels() != 1) {
    throw std::invalid_argument(std::string("mapping ") + modality
                                + " image must be single-channel");
  }
  if (remap.identity) return;
  const cv::Mat empty_fractional_map;
  const cv::Mat& fractional_map =
      interpolation == cv::INTER_NEAREST && remap.integerCoordinates
          ? empty_fractional_map : remap.map2;
  cv::Mat allocated_output;
  cv::remap(image, allocated_output, remap.map1, fractional_map, interpolation,
            cv::BORDER_CONSTANT, cv::Scalar::all(0));
  image = std::move(allocated_output);
}

bool MappingImagePreprocessor::process(const size_t cameraIndex,
                                       CameraMeasurement& measurement,
                                       const size_t downsampling) const {
  if (!config_.enabled) return true;
  if (cameraIndex != config_.targetCameraIndex) return false;
  const uint64_t revision = geometryRevision(downsampling);

  const auto validateRevision = [revision](const cv::Mat& image,
                                            const uint64_t current_revision,
                                            const char* modality) {
    if (!image.empty() && current_revision != 0
        && current_revision != revision) {
      throw std::invalid_argument(std::string("mapping ") + modality
                                  + " was already converted to another geometry");
    }
  };
  validateRevision(measurement.measurement.image,
                   measurement.measurement.imageGeometryRevision, "RGB");
  validateRevision(measurement.measurement.depthImage,
                   measurement.measurement.depthGeometryRevision, "depth");

  const auto convertedOrEmpty = [revision](const cv::Mat& image,
                                            const uint64_t current_revision) {
    return image.empty() || current_revision == revision;
  };
  if (convertedOrEmpty(measurement.measurement.image,
                       measurement.measurement.imageGeometryRevision)
      && convertedOrEmpty(measurement.measurement.depthImage,
                          measurement.measurement.depthGeometryRevision)) {
    return true;
  }

  const auto cache = factorCache(downsampling);

  auto processTagged = [&](cv::Mat& image,
                           uint64_t& current_revision,
                           std::shared_ptr<FrameBufferLifetime>& source_lifetime,
                           const Remap& remap,
                           const int interpolation,
                           const char* modality) {
    if (image.empty()) return;
    if (current_revision == revision) return;
    processMat(remap, image, interpolation, modality);
    if (!remap.identity) {
      source_lifetime.reset();
    }
    current_revision = revision;
  };

  processTagged(measurement.measurement.image,
                measurement.measurement.imageGeometryRevision,
                measurement.measurement.imageBufferLifetime,
                *cache->rgb, cv::INTER_LINEAR, "RGB");
  processTagged(measurement.measurement.depthImage,
                measurement.measurement.depthGeometryRevision,
                measurement.measurement.depthBufferLifetime,
                *cache->depth, cv::INTER_NEAREST, "depth");

  return true;
}

}  
