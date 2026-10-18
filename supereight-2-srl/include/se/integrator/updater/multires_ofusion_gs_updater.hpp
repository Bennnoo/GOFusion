/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2019-2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2019-2021 Nils Funk
 * SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#ifndef SE_MULTIRES_OFUSION_GS_UPDATER_HPP
#define SE_MULTIRES_OFUSION_GS_UPDATER_HPP

#include <se/integrator/updater/multires_ofusion_core.hpp>
#include <gs/rgb_ingest.cuh>
#include <gs/geometry_supervision.cuh>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>


namespace se {

// Multi-res Occupancy updater (GSFusion++)
template<Colour ColB, int BlockSize, typename SensorT>
class GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>, SensorT> {
    public:
    typedef Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize> MapType;
    typedef typename MapType::DataType DataType;
    typedef typename MapType::OctreeType OctreeType;
    typedef typename MapType::NodeType NodeType;
    typedef typename MapType::BlockType BlockType;

    struct UpdaterConfig {
        UpdaterConfig(const MapType& map) :
                sigma_min(map.getRes() * map.getDataConfig().field.sigma_min_factor),
                sigma_max(map.getRes() * map.getDataConfig().field.sigma_max_factor),
                tau_min(map.getRes() * map.getDataConfig().field.tau_min_factor),
                tau_max(map.getRes() * map.getDataConfig().field.tau_max_factor)
        {
        }

        const float sigma_min;
        const float sigma_max;
        const float tau_min;
        const float tau_max;
    };

    GSUpdater(MapType& map,
              const timestamp_t timestamp,
              const Measurements<SensorT>& measurements,
              gs::GaussianModel& gs_model,
              std::vector<gs::Camera>& gs_cam_list,
              std::vector<torch::Tensor>& gt_img_list,
              std::vector<gs::GeometryTarget>& geometry_target_list,
              std::vector<torch::Tensor>& depth_target_list_cpu,
              gs::DataQueue& data_queue,
              std::vector<gs::Camera>* gs_cam_list_cpu = nullptr,
              std::vector<torch::Tensor>* gt_img_list_cpu = nullptr,
              gs::PoseUpdateVector* pose_updates = nullptr,
              gs::FisherTrainingSampleVector* fisher_training_samples = nullptr,
              gs::DataPacket* preview_packet = nullptr,
              bool* occupancy_completed = nullptr);


    void operator()(VolumeCarverAllocation& allocation_list,
                    std::vector<const OctantBase*>* updated_octants = nullptr);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    void propagateToRoot(std::vector<OctantBase*>& block_list);

    void freeBlock(OctantBase* octant_ptr);

    void updateBlock(OctantBase* octant_ptr, VarianceState variance_state, bool project_inside);


    void freeNodeRecurse(OctantBase* octant_ptr,
                         int depth,
                         std::vector<std::vector<OctantBase*>>& node_set_local,
                         std::vector<OctantBase*>& freed_blocks_local);


    void updateGSModel(std::vector<gs::Point>& positions,
                       std::vector<gs::Color>& colors,
                       std::vector<gs::Point>& scales_xyz,
                       std::vector<gs::Quat>& rotations_wxyz);

    void syncCameraCpuEntry(size_t sample_index, const gs::Camera& camera);
    void recordJointPoseUpdate(size_t sample_index,
                               bool is_current_frame,
                               const gs::Camera& camera);

    private:
    MapType& map_;
    OctreeType& octree_;
    const SensorT& sensor_;
    const Image<float>& depth_img_;
    // const Image<float>* const sigma_img_;
    const Eigen::Isometry3f T_CW_;
    const SensorT* const colour_sensor_;
    const Image<colour_t>* const colour_img_;
    Eigen::Isometry3f T_CcC_;
    const bool has_colour_;
    const timestamp_t timestamp_;
    const float map_res_;
    const UpdaterConfig config_;
    std::vector<std::vector<OctantBase*>> node_set_;
    std::vector<OctantBase*> freed_block_list_;

    std::set<OctantBase*> updated_octants_;
    bool track_updated_octants_ = false;

    gs::GaussianModel& gs_model_;
    std::vector<gs::Camera>& gs_cam_list_;
    std::vector<torch::Tensor>& gt_img_list_;
    std::vector<gs::GeometryTarget>& geometry_target_list_;
    std::vector<torch::Tensor>& depth_target_list_cpu_;
    std::vector<gs::Camera>* gs_cam_list_cpu_ = nullptr;
    std::vector<torch::Tensor>* gt_img_list_cpu_ = nullptr;
    gs::PoseUpdateVector* pose_updates_ = nullptr;
    gs::FisherTrainingSampleVector* fisher_training_samples_ = nullptr;
    gs::DataPacket* preview_packet_ = nullptr;
    gs::DataQueue& data_queue_;
    bool* occupancy_completed_ = nullptr;

    gs::DataPacket data_packet_;
    gs::Camera cur_gs_cam_;
    gs::RgbInputView rgb_input_;
    torch::Tensor cur_gt_img_;
    torch::Tensor cur_gt_img_host_cpu_;
    torch::Tensor cur_depth_target_cpu_;
    gs::GeometryTarget cur_geometry_target_;
    bool isKeyframe_ = false;
    double start_time_;
    double end_time_;
    Eigen::Isometry3f T_WCc_;
    float colour_fx_ = 0.f;
    float colour_fy_ = 0.f;
    float colour_cx_ = 0.f;
    float colour_cy_ = 0.f;
    float colour_scale_u_ = 1.f;
    float colour_scale_v_ = 1.f;
};

} // namespace se

#include "impl/multires_ofusion_gs_updater_impl.hpp"

#endif // SE_MULTIRES_OFUSION_GS_UPDATER_HPP
