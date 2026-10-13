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
 * @file ceres/GpsErrorSynchronous.hpp
 * @brief Header file for the GpsErrorSynchronous class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CERES_GPSERRORSYNCHRONOUS_HPP_
#define INCLUDE_OKVIS_CERES_GPSERRORSYNCHRONOUS_HPP_

#include <vector>
#include <memory>
#include <ceres/sized_cost_function.h>
#include <ceres/covariance.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/Parameters.hpp>

namespace okvis {
namespace ceres {

class GpsErrorSynchronous :
    public ::ceres::SizedCostFunction<3 ,
        7 ,
        7 >,
    public ErrorInterface {

 public:

  typedef ::ceres::SizedCostFunction<3, 7, 7> base_t;

  static const int kNumResiduals = 3;

  typedef Eigen::Matrix<double, 3, 3> covariance_t;

  typedef Eigen::Vector3d measurement_t;

  typedef covariance_t information_t;

  typedef Eigen::Matrix<double, 3, 7> jacobian0_t;

  typedef Eigen::Matrix<double, 3, 7> jacobian1_t;

  GpsErrorSynchronous();

  GpsErrorSynchronous(uint64_t cameraId, const measurement_t & measurement,
                    const covariance_t & information, const GpsParameters & gpsParameters);

  virtual ~GpsErrorSynchronous()
  {
  }

  virtual void setGpsParameters(const GpsParameters & gpsParameters){
      gpsParameters_ = gpsParameters;
  }

  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  virtual void setInformation(const covariance_t& information);

  virtual void setCameraId(uint64_t cameraId) {
    cameraId_ = cameraId;
  }

  uint64_t cameraId() const {
    return cameraId_;
  }

  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  virtual const covariance_t& information() const
  {
    return information_;
  }

  virtual const covariance_t& covariance() const
  {
    return covariance_;
  }

  virtual const okvis::GpsParameters& gpsParameters() const {
    return gpsParameters_;
  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  virtual bool VerifyJacobianNumDiff(double const* const * parameters, double** jacobian) const;

  int residualDim() const
  {
    return kNumResiduals;
  }

  int parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  int parameterBlockDim(int parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  virtual std::string typeInfo() const
  {
    return "GpsErrorSynchronous";
  }

 protected:

  measurement_t measurement_; 

  GpsParameters gpsParameters_; 

  covariance_t information_; 
  covariance_t squareRootInformation_; 
  covariance_t covariance_; 

  uint64_t cameraId_; 

};

}  
}  

#endif 
