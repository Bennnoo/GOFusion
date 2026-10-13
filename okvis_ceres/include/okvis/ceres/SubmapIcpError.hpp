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
 * @file SubmapIcpError.hpp
 * @brief Header file for the Submap ICP error class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CERES_SUBMAPICPERROR_HPP_
#define INCLUDE_OKVIS_CERES_SUBMAPICPERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>

#include <okvis/mapTypedefs.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <se/supereight.hpp>
#include <Eigen/StdVector>

namespace okvis {

namespace ceres {

class SubmapIcpError : public ::ceres::SizedCostFunction<
        1 ,
        7, 
        7 >, public ErrorInterface {

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

    typedef ::ceres::SizedCostFunction<1, 7, 7> base_t;

    static const int kNumResiduals = 1;

    typedef double information_t;

    typedef double covariance_t;

    SubmapIcpError() = delete;

    SubmapIcpError(

            const SupereightMapType& submap,
            const Eigen::Vector3d& point_b,
            const double sigma_measurement);

    virtual ~SubmapIcpError() override = default;

    void setPb(const Eigen::Vector3d& point_b);

    const SupereightMapType& submap() const {

      return submapPtr_;
    }

    const Eigen::Vector3d& p_b() const {
      return p_b_;
    }

    virtual bool Evaluate(double const* const * parameters, double* residuals,
                          double** jacobians) const override final;

    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals, double** jacobians,
                                      double** jacobiansMinimal) const override final;

    int residualDim() const override final {
      return kNumResiduals;
    }

    int parameterBlocks() const override final {
      return int(parameter_block_sizes().size());
    }

    int parameterBlockDim(int parameterBlockId) const override final {
      return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
    }

    virtual std::string typeInfo() const override final {
      return "SubmapIcpError";
    }

    std::pair<bool, bool> nonZeroFieldAndGradient(const double *const *parameters, float& fieldVal, float& gradNorm);

protected:

    const SupereightMapType& submapPtr_;

    Eigen::Vector3d p_b_; 

    covariance_t sigma_measurement_; 
    mutable covariance_t sigma_map_; 
    mutable information_t squareRootInformation_; 

};

}  
}  

#endif 
