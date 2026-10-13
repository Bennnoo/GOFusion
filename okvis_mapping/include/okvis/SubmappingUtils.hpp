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

#ifndef INCLUDE_OKVIS_SUBMAPPINGUTILS_HPP
#define INCLUDE_OKVIS_SUBMAPPINGUTILS_HPP

#include <se/supereight.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/mapTypedefs.hpp>

namespace okvis{

    template<se::Safe SafeB>
    std::optional<se::field_t> interpFieldMeanOccup(
        const SupereightMapType& map,
        const Eigen::Vector3f& point_W,
        const se::Scale desired_scale = 0,
        se::Scale *const returned_scale = nullptr)
    {
        auto valid = [](const typename SupereightMapType::DataType& d) { return is_valid(d); };
        auto get = [](const typename SupereightMapType::DataType& d) { return d.field.occupancy; };
        return map.interp<decltype(valid), decltype(get), SafeB>(
            point_W,
            valid,
            get,
            desired_scale,
            returned_scale
        );
    }

    template<se::Safe SafeB>
    std::optional<se::field_vec_t> gradFieldMeanOccup(
        const SupereightMapType& map,
        const Eigen::Vector3f& point_W,
        const se::Scale desired_scale = 0,
        se::Scale *const returned_scale = nullptr)
    {

        auto valid = [](const typename SupereightMapType::DataType& d) { return is_valid(d); };
        auto get = [](const typename SupereightMapType::DataType& d) { return d.field.occupancy; };

        return map.grad<decltype(valid), decltype(get), SafeB>(
            point_W,
            valid,
            get,
            desired_scale,
            returned_scale
        );
    }

    void save_bounding_box_vtk(const se::Submap<okvis::SupereightMapType>& submap, const std::string& filename);

    void downsamplePointCloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& originalPointCloud,
                              std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downsampledPointCloud,
                              size_t num_of_points);

    void downsamplePointsUncertainty(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points, const std::vector<float>& sigmas,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downPoints, std::vector<float>& downSigma, size_t maxNum);

    class LidarMotionCompensation : Propagator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LidarMotionCompensation(const Time & t_start, const ImuMeasurementDeque & imuMeasurements);

        bool appendTo(
                const ImuMeasurementDeque & imuMeasurements,
                const kinematics::Transformation& T_WS,
                const SpeedAndBias & speedAndBiases, const Time & t_end);

        bool getState(
                const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
                kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
                Eigen::Vector3d & omega_S);

    private:

        ImuMeasurementDeque::const_iterator imu_iterator_;
        bool hasStarted_ = false;

        Eigen::Quaterniond Delta_q_interp_ = Eigen::Quaterniond(1,0,0,0); 
        Eigen::Matrix3d C_integral_interp_ = Eigen::Matrix3d::Zero(); 
        Eigen::Matrix3d C_doubleintegral_interp_ = Eigen::Matrix3d::Zero(); 
        Eigen::Vector3d acc_integral_interp_ = Eigen::Vector3d::Zero(); 
        Eigen::Vector3d acc_doubleintegral_interp_ = Eigen::Vector3d::Zero(); 

        Eigen::Matrix3d cross_interp_ = Eigen::Matrix3d::Zero(); 

        Eigen::Matrix3d dalpha_db_g_interp_ = Eigen::Matrix3d::Zero(); 
        Eigen::Matrix3d dv_db_g_interp_ = Eigen::Matrix3d::Zero(); 
        Eigen::Matrix3d dp_db_g_interp_ = Eigen::Matrix3d::Zero(); 

        Eigen::Vector3d omega_S_interp_ = Eigen::Vector3d::Zero(); 

    };

}

#endif 
