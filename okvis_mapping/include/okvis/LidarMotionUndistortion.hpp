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

#ifndef INCLUDE_OKVIS_LIDARMOTIONUNDISTORTION_HPP
#define INCLUDE_OKVIS_LIDARMOTIONUNDISTORTION_HPP

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/mapTypedefs.hpp>
#include <se/map/map.hpp>

namespace okvis{
    class LidarMotionUndistortion{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LidarMotionUndistortion() = delete;

        LidarMotionUndistortion(LidarMotionUndistortion && ) = default;

        LidarMotionUndistortion(const LidarMotionUndistortion & ) = default;

        LidarMotionUndistortion(const State& initialState, const kinematics::Transformation& T_WS_live,
                                const kinematics::Transformation& T_SL,
                                const LidarMeasurementDeque& lidarMeasurementDeque,
                                const ImuMeasurementDeque& imuMeasurementDeque);

        bool deskew();

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> & deskewedPointCloud(){
          return lidarScanDeskewed_;
        }

        bool downsample(size_t num_output_points, double voxel_grid_resolution=0.1);

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& deskewedDownsampledPointCloud(){
          return lidarScanDeskewedDownsampled_;
        }

        size_t filterObserved(const SupereightMapType* map, const okvis::kinematics::Transformation& T_WM);

    private:

        State initialState_; 
        kinematics::Transformation T_WS_live_; 
        kinematics::Transformation T_SW_live_; 
        kinematics::Transformation T_SL_; 
        LidarMeasurementDeque lidarScan_;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> lidarScanDeskewed_;

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> lidarScanDeskewedDownsampled_;

        ImuMeasurementDeque imuMeasurementDeque_;

    };
}

#endif 
