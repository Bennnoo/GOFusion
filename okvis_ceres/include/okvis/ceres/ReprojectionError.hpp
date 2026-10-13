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
 * @file ceres/ReprojectionError.hpp
 * @brief Header file for the ReprojectionError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_REPROJECTIONERROR_HPP_
#define INCLUDE_OKVIS_CERES_REPROJECTIONERROR_HPP_

#include <memory>

#include <ceres/sized_cost_function.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>

namespace okvis {
namespace ceres {

template<class GEOMETRY_TYPE>
class ReprojectionError : public ReprojectionError2dBase
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef GEOMETRY_TYPE camera_geometry_t;

  typedef ::ceres::SizedCostFunction<2, 7, 4, 7> base_t;

  static const int kNumResiduals = 2;

  typedef Eigen::Vector2d keypoint_t;

  ReprojectionError();

  ReprojectionError(std::shared_ptr<const camera_geometry_t> cameraGeometry,
                    uint64_t cameraId, const measurement_t & measurement,
                    const covariance_t & information);

  virtual ~ReprojectionError() override = default;

  virtual void setMeasurement(const measurement_t& measurement) override final
  {
    measurement_ = measurement;
  }

  void setCameraGeometry(
      std::shared_ptr<const camera_geometry_t> cameraGeometry)
  {
    cameraGeometry_ = cameraGeometry;
  }

  virtual void setInformation(const covariance_t& information) override final;

  virtual const measurement_t& measurement() const override final
  {
    return measurement_;
  }

  virtual const covariance_t& information() const override final
  {
    return information_;
  }

  virtual const covariance_t& covariance() const override final
  {
    return covariance_;
  }

  virtual std::shared_ptr<ReprojectionError2dBase> clone() const override final {
    return std::shared_ptr<ReprojectionError2dBase>(
          new ReprojectionError<camera_geometry_t>(
            cameraGeometry_ ,cameraId_, measurement_, information_));
  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const override final;

  int residualDim() const override final
  {
    return kNumResiduals;
  }

  int parameterBlocks() const override final
  {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final
  {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::string typeInfo() const override final
  {
    return "ReprojectionError";
  }

 protected:

  measurement_t measurement_; 

  std::shared_ptr<const camera_geometry_t> cameraGeometry_;

  covariance_t information_; 
  covariance_t squareRootInformation_; 
  covariance_t covariance_; 

};

}  
}  

#include "implementation/ReprojectionError.hpp"

#endif 
