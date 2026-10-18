/*
 * SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2020-2021 Nils Funk
 * SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#ifndef SE_VOXEL_CARVER_IMPL_HPP
#define SE_VOXEL_CARVER_IMPL_HPP

#include <array>

namespace se {

template<se::Colour ColB, int BlockSize, typename SensorT>
VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
             SensorT>::VolumeCarver(MapType& map,
                                    const SensorT& sensor,
                                    const se::Image<float>& depth_img,
                                    const Eigen::Isometry3f& T_WS,
                                    const timestamp_t /* timestamp */) :
        map_(map),
        octree_(map.getOctree()),
        sensor_(sensor),
        depth_pooling_img_(depth_img),
        T_SW_(T_WS.inverse()),
        map_res_(map.getRes()),
        config_(map),
        max_depth_value_(
            std::min(sensor.far_plane, depth_pooling_img_.maxValue() + config_.tau_max)),
        zero_depth_band_(1.0e-6f),
        size_to_radius_(std::sqrt(3.0f) / 2.0f)
{
}

template<se::Colour ColB, int BlockSize, typename SensorT>
VolumeCarverAllocation
VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
             SensorT>::operator()()
{
    constexpr int kRootChildren = 8;
    allocation_list_.node_list.clear();
    allocation_list_.block_list.clear();
    allocation_list_.variance_state_list.clear();
    allocation_list_.projects_inside_list.clear();

    NodeType* root_ptr = static_cast<NodeType*>(octree_.getRoot());
    const int child_size = root_ptr->size / 2;
    octree_.allocateChildren(root_ptr);

    // One local allocation list per root child (no locks needed).
    std::array<VolumeCarverAllocation, kRootChildren> local_lists;
    for (auto& L : local_lists) {
        L.node_list.clear();
        L.block_list.clear();
        L.variance_state_list.clear();
        L.projects_inside_list.clear();
    }

#pragma omp parallel for
    for (int child_idx = 0; child_idx < kRootChildren; ++child_idx) {
        (*this)(root_ptr->getChildCoord(child_idx),
                child_size,
                1,
                root_ptr->getChild(child_idx),
                local_lists[child_idx]);
    }

    // Merge local lists.
    size_t total_nodes = 0;
    size_t total_blocks = 0;
    for (int i = 0; i < kRootChildren; ++i) {
        total_nodes += local_lists[i].node_list.size();
        total_blocks += local_lists[i].block_list.size();
    }

    allocation_list_.node_list.reserve(total_nodes);
    allocation_list_.block_list.reserve(total_blocks);
    allocation_list_.variance_state_list.reserve(total_blocks);
    allocation_list_.projects_inside_list.reserve(total_blocks);

    for (int i = 0; i < kRootChildren; ++i) {
        auto& L = local_lists[i];

        if (!L.node_list.empty()) {
            allocation_list_.node_list.insert(allocation_list_.node_list.end(),
                                              L.node_list.begin(),
                                              L.node_list.end());
        }

        if (!L.block_list.empty()) {
            allocation_list_.block_list.insert(allocation_list_.block_list.end(),
                                               L.block_list.begin(),
                                               L.block_list.end());
            allocation_list_.variance_state_list.insert(allocation_list_.variance_state_list.end(),
                                                        L.variance_state_list.begin(),
                                                        L.variance_state_list.end());
            allocation_list_.projects_inside_list.insert(allocation_list_.projects_inside_list.end(),
                                                         L.projects_inside_list.begin(),
                                                         L.projects_inside_list.end());
        }
    }

    // Extend octree AABB.
    for (const OctantBase* octant : allocation_list_.node_list) {
        if (octant->isLeaf()) {
            octree_.aabbExtend(octant->coord, octant->size);
        }
    }

    return allocation_list_;
}

template<se::Colour ColB, int BlockSize, typename SensorT>
bool VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>, SensorT>::
    crossesFrustum(std::vector<srl::projection::ProjectionStatus>& proj_corner_stati)
{
    for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
        if (proj_corner_stati[corner_idx] == srl::projection::ProjectionStatus::Successful) {
            return true;
        }
    }
    return false;
}

template<se::Colour ColB, int BlockSize, typename SensorT>
bool VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
                  SensorT>::cameraInNode(const Eigen::Vector3i& node_coord,
                                         const int node_size,
                                         const Eigen::Isometry3f& T_WS)
{
    assert(node_size > 0);
    Eigen::Vector3f voxel_coord_f;
    map_.pointToVoxel(T_WS.translation(), voxel_coord_f);
    const Eigen::Array3f lower = node_coord.cast<float>().array();
    const Eigen::Array3f upper = lower + node_size;
    return (voxel_coord_f.array() >= lower).all() && (voxel_coord_f.array() < upper).all();
}

template<se::Colour ColB, int BlockSize, typename SensorT>
se::VarianceState
VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
             SensorT>::computeVariance(const float depth_value_min,
                                       const float depth_value_max,
                                       const float node_dist_min_m,
                                       const float node_dist_max_m,
                                       const float std_dev_max)
{
    assert(depth_value_min <= depth_value_max);
    assert(node_dist_min_m <= node_dist_max_m);
    // Assume worst case scenario -> no multiplication with proj_scale
    const float z_diff_max = (node_dist_max_m - depth_value_min); // * proj_scale;
    const float z_diff_min = (node_dist_min_m - depth_value_max); // * proj_scale;

    const float tau_max =
        compute_tau(depth_value_max, config_.tau_min, config_.tau_max, map_.getDataConfig());
    const float three_sigma_min = 3.0f * std_dev_max;

    if (z_diff_min > 1.25 * tau_max) { // behind of surface
        return se::VarianceState::Undefined;
    }
    else if (z_diff_max < -1.25 * three_sigma_min) { // guranteed free space
        return se::VarianceState::Constant;
    }
    else {
        return se::VarianceState::Gradient;
    }
}

template<se::Colour ColB, int BlockSize, typename SensorT>
template<class SensorTDummy>
typename std::enable_if_t<std::is_same<SensorTDummy, PinholeCamera>::value, void>
VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
             SensorT>::operator()(const Eigen::Vector3i& octant_coord,
                                  const int octant_size,
                                  const int octant_depth,
                                  se::OctantBase* octant_ptr,
                                  VolumeCarverAllocation& allocation_list_local)
{
    struct Task {
        Eigen::Vector3i coord;
        int size;
        int depth;
        se::OctantBase* ptr;
    };

    static thread_local std::vector<Task> stack;
    stack.clear();
    if (stack.capacity() < 512) {
        stack.reserve(512);
    }
    stack.push_back(Task{octant_coord, octant_size, octant_depth, octant_ptr});

    static thread_local std::vector<srl::projection::ProjectionStatus> proj_node_corner_stati;
    if (proj_node_corner_stati.capacity() < 8) {
        proj_node_corner_stati.reserve(8);
    }

    while (!stack.empty()) {
        const Task task = stack.back();
        stack.pop_back();

        const Eigen::Vector3i& oct_coord = task.coord;
        const int oct_size = task.size;
        const int oct_depth = task.depth;
        se::OctantBase* oct_ptr = task.ptr;

        assert(oct_size > 0);
        assert(oct_depth >= 0);
        assert(oct_ptr);

        Eigen::Vector3f node_centre_point_W;
        map_.voxelToPoint(oct_coord, oct_size, node_centre_point_W);

        const Eigen::Vector3f node_centre_point_S = T_SW_ * node_centre_point_W;

        const float approx_depth_value_max =
            node_centre_point_S.z() + oct_size * size_to_radius_ * map_res_;
        const float approx_depth_value_min =
            node_centre_point_S.z() - oct_size * size_to_radius_ * map_res_;

        if (approx_depth_value_min > max_depth_value_
            || approx_depth_value_max < zero_depth_band_) {
            continue;
        }

        Eigen::Matrix<float, 3, 8> node_corner_points_W;
        map_.voxelToCornerPoints(oct_coord, oct_size, node_corner_points_W);
        Eigen::Matrix<float, 3, 8> node_corner_points_S = T_SW_ * node_corner_points_W;

        Eigen::VectorXi node_corners_infront(8);
        node_corners_infront << 1, 1, 1, 1, 1, 1, 1, 1;
        for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
            if (node_corner_points_S(2, corner_idx) < zero_depth_band_) {
                node_corners_infront(corner_idx) = 0;
            }
        }

        const int num_node_corners_infront = node_corners_infront.sum();
        if (num_node_corners_infront == 0) {
            continue;
        }

        Eigen::Matrix2Xf proj_node_corner_pixels_f(2, 8);
        const Eigen::VectorXf node_corners_diff = node_corner_points_S.row(2);
        proj_node_corner_stati.clear();
        sensor_.model.projectBatch(
            node_corner_points_S, &proj_node_corner_pixels_f, &proj_node_corner_stati);

        bool should_split = false;
        bool projects_inside = false;
        se::VarianceState variance_state = se::VarianceState::Gradient;
        se::Pixel pooling_pixel = se::Pixel::crossingUnknownPixel();

        if (oct_depth < octree_.getBlockDepth() + 1) {
            if (num_node_corners_infront < 8) {
                if (cameraInNode(oct_coord, oct_size, T_SW_.inverse())) {
                    should_split = true;
                }
                else if (crossesFrustum(proj_node_corner_stati)) {
                    should_split = true;
                }
                else if (sensor_.sphereInFrustumInf(node_centre_point_S,
                                                    oct_size * size_to_radius_ * map_res_)) {
                    should_split = true;
                }
                else {
                    continue;
                }
            }
            else {
                proj_node_corner_pixels_f = (proj_node_corner_pixels_f.array() > 1e7f)
                                                .select(1e7f, proj_node_corner_pixels_f.array());
                proj_node_corner_pixels_f = (proj_node_corner_pixels_f.array() < -1e7f)
                                                .select(-1e7f, proj_node_corner_pixels_f.array());

                const Eigen::Vector2i img_bb_min =
                    proj_node_corner_pixels_f.rowwise().minCoeff().cast<int>();
                const Eigen::Vector2i img_bb_max =
                    proj_node_corner_pixels_f.rowwise().maxCoeff().cast<int>();
                const float node_dist_min_m = node_corners_diff.minCoeff();
                const float node_dist_max_m = node_corners_diff.maxCoeff();

                pooling_pixel = depth_pooling_img_.conservativeQuery(img_bb_min, img_bb_max);

                if (pooling_pixel.status_known == se::Pixel::statusKnown::unknown) {
                    continue;
                }

                if (node_dist_min_m > pooling_pixel.max + config_.tau_max) {
                    continue;
                }

                const float depth_max = pooling_pixel.max;
                const auto& field_cfg = map_.getDataConfig().field;
                float std_dev_max = 0.0f;
                if (field_cfg.uncertainty_model == se::UncertaintyModel::Linear) {
                    std_dev_max = field_cfg.k_sigma * depth_max;
                }
                else {
                    std_dev_max = field_cfg.k_sigma * depth_max * depth_max;
                }
                std_dev_max = std::min(std::max(std_dev_max, config_.sigma_min), config_.sigma_max);

                variance_state = computeVariance(pooling_pixel.min,
                                                 pooling_pixel.max,
                                                 node_dist_min_m,
                                                 node_dist_max_m,
                                                 std_dev_max);

                if (variance_state != se::VarianceState::Gradient) {
                    typename OctreeType::DataType child_data = (oct_ptr->is_block)
                        ? static_cast<BlockType*>(oct_ptr)->maxData()
                        : static_cast<NodeType*>(oct_ptr)->data();

                    if (child_data.field.observed
                        && get_field(child_data)
                            <= 0.95 * MapType::DataType::FieldType::min_occupancy) {
                        continue;
                    }
                }

                if (pooling_pixel.status_crossing == se::Pixel::statusCrossing::crossing) {
                    should_split = true;
                }
                else if (pooling_pixel.status_known == se::Pixel::statusKnown::part_known) {
                    should_split = true;
                }
                else if (variance_state == se::VarianceState::Gradient) {
                    should_split = true;
                }

                projects_inside = (pooling_pixel.status_known == se::Pixel::known);
            }
        }

        if (should_split) {
            if (oct_ptr->is_block) {
                allocation_list_local.block_list.push_back(oct_ptr);
                allocation_list_local.variance_state_list.push_back(variance_state);
                allocation_list_local.projects_inside_list.push_back(projects_inside);
            }
            else {
                NodeType* node_ptr = static_cast<NodeType*>(oct_ptr);
                const int child_size = node_ptr->size / 2;
                octree_.allocateChildren(node_ptr);
                for (int child_idx = 0; child_idx < 8; ++child_idx) {
                    stack.push_back(Task{node_ptr->getChildCoord(child_idx),
                                         child_size,
                                         oct_depth + 1,
                                         node_ptr->getChild(child_idx)});
                }
            }
        }
        else {
            assert(oct_depth);

            if (oct_ptr->is_block) {
                allocation_list_local.block_list.push_back(oct_ptr);
                allocation_list_local.variance_state_list.push_back(variance_state);
                allocation_list_local.projects_inside_list.push_back(projects_inside);
            }
            else if (variance_state == se::VarianceState::Constant) {
                allocation_list_local.node_list.push_back(oct_ptr);
            }
        }
    }
}

template<se::Colour ColB, int BlockSize, typename SensorT>
template<class SensorTDummy>
typename std::enable_if_t<std::is_same<SensorTDummy, se::RangeImageLidar>::value, void>
VolumeCarver<Map<Data<se::Field::Occupancy, ColB>, se::Res::Multi, BlockSize>,
             SensorT>::operator()(const Eigen::Vector3i& octant_coord,
                                  const int octant_size,
                                  const int octant_depth,
                                  se::OctantBase* octant_ptr,
                                  VolumeCarverAllocation& allocation_list_local)
{
    struct Task {
        Eigen::Vector3i coord;
        int size;
        int depth;
        se::OctantBase* ptr;
    };

    static thread_local std::vector<Task> stack;
    stack.clear();
    if (stack.capacity() < 512) {
        stack.reserve(512);
    }
    stack.push_back(Task{octant_coord, octant_size, octant_depth, octant_ptr});

    static thread_local std::vector<srl::projection::ProjectionStatus> proj_node_corner_stati;
    if (proj_node_corner_stati.capacity() < 8) {
        proj_node_corner_stati.reserve(8);
    }

    while (!stack.empty()) {
        const Task task = stack.back();
        stack.pop_back();

        const Eigen::Vector3i& oct_coord = task.coord;
        const int oct_size = task.size;
        const int oct_depth = task.depth;
        se::OctantBase* oct_ptr = task.ptr;

        assert(oct_size > 0);
        assert(oct_depth >= 0);
        assert(oct_ptr);

        Eigen::Vector3f node_centre_point_W;
        map_.voxelToPoint(oct_coord, oct_size, node_centre_point_W);

        const Eigen::Vector3f node_centre_point_S = T_SW_ * node_centre_point_W;

        const float approx_depth_value_max =
            node_centre_point_S.norm() + oct_size * size_to_radius_ * map_res_;
        const float approx_depth_value_min =
            node_centre_point_S.norm() - oct_size * size_to_radius_ * map_res_;

        if (approx_depth_value_min > max_depth_value_) {
            continue;
        }

        bool should_split = false;
        bool projects_inside = false;
        se::VarianceState variance_state = se::VarianceState::Gradient;
        se::Pixel pooling_pixel = se::Pixel::crossingUnknownPixel();

        if (oct_depth < octree_.getBlockDepth() + 1) {
            if (cameraInNode(oct_coord, oct_size, T_SW_.inverse())) {
                should_split = true;
            }
            else {
                Eigen::Matrix<float, 3, 8> node_corner_points_W;
                map_.voxelToCornerPoints(oct_coord, oct_size, node_corner_points_W);
                Eigen::Matrix<float, 3, 8> node_corner_points_S = T_SW_ * node_corner_points_W;

                Eigen::Matrix2Xf proj_node_corner_pixels_f(2, 8);
                proj_node_corner_stati.clear();
                sensor_.model.projectBatch(
                    node_corner_points_S, &proj_node_corner_pixels_f, &proj_node_corner_stati);

                Eigen::Vector2f proj_node_centre_pixel_f;
                sensor_.model.project(node_centre_point_S, &proj_node_centre_pixel_f);

                float proj_node_centre_pixel_u_f = proj_node_centre_pixel_f.x();
                float proj_node_centre_pixel_u_comp_f =
                    proj_node_centre_pixel_u_f + sensor_.model.imageWidth() / 2;

                if (proj_node_centre_pixel_u_comp_f > sensor_.model.imageWidth() - 0.5) {
                    proj_node_centre_pixel_u_comp_f =
                        proj_node_centre_pixel_u_comp_f - sensor_.model.imageWidth();
                }

                Eigen::VectorXf proj_node_corner_pixels_u_f(8);
                for (int i = 0; i < 8; i++) {
                    if (proj_node_centre_pixel_u_f < proj_node_centre_pixel_u_comp_f
                        && proj_node_corner_pixels_f(0, i) > proj_node_centre_pixel_u_f
                        && proj_node_corner_pixels_f(0, i) > proj_node_centre_pixel_u_comp_f) {
                        proj_node_corner_pixels_u_f(i) =
                            proj_node_corner_pixels_f(0, i) - sensor_.model.imageWidth();
                    }
                    else if (proj_node_centre_pixel_u_f > proj_node_centre_pixel_u_comp_f
                             && proj_node_corner_pixels_f(0, i) < proj_node_centre_pixel_u_f
                             && proj_node_corner_pixels_f(0, i) < proj_node_centre_pixel_u_comp_f) {
                        proj_node_corner_pixels_u_f(i) =
                            proj_node_corner_pixels_f(0, i) + sensor_.model.imageWidth();
                    }
                    else {
                        proj_node_corner_pixels_u_f(i) = proj_node_corner_pixels_f(0, i);
                    }
                }
                Eigen::VectorXf proj_node_corner_pixels_v_f = proj_node_corner_pixels_f.row(1);

                int u_min = proj_node_corner_pixels_u_f.minCoeff();
                int u_max = proj_node_corner_pixels_u_f.maxCoeff();
                int v_min = proj_node_corner_pixels_v_f.minCoeff();
                int v_max = proj_node_corner_pixels_v_f.maxCoeff();

                const Eigen::Vector2i img_bb_min(u_min, v_min);
                const Eigen::Vector2i img_bb_max(u_max, v_max);

                pooling_pixel = depth_pooling_img_.conservativeQuery(img_bb_min, img_bb_max);

                if (pooling_pixel.status_known == se::Pixel::statusKnown::unknown) {
                    continue;
                }

                if (approx_depth_value_min > pooling_pixel.max + config_.tau_max) {
                    continue;
                }

                const float depth_max = pooling_pixel.max;
                const auto& field_cfg = map_.getDataConfig().field;
                float std_dev_max = 0.0f;
                if (field_cfg.uncertainty_model == se::UncertaintyModel::Linear) {
                    std_dev_max = field_cfg.k_sigma * depth_max;
                }
                else {
                    std_dev_max = field_cfg.k_sigma * depth_max * depth_max;
                }
                std_dev_max = std::min(std::max(std_dev_max, config_.sigma_min), config_.sigma_max);

                variance_state = computeVariance(pooling_pixel.min,
                                                 pooling_pixel.max,
                                                 approx_depth_value_min,
                                                 approx_depth_value_max,
                                                 std_dev_max);

                if (variance_state != se::VarianceState::Gradient) {
                    typename OctreeType::DataType child_data = (oct_ptr->is_block)
                        ? static_cast<BlockType*>(oct_ptr)->maxData()
                        : static_cast<NodeType*>(oct_ptr)->data();

                    if (child_data.field.observed
                        && get_field(child_data)
                            <= 0.95 * MapType::DataType::FieldType::min_occupancy) {
                        continue;
                    }
                }

                if (pooling_pixel.status_crossing == se::Pixel::statusCrossing::crossing) {
                    should_split = true;
                }
                else if (pooling_pixel.status_known == se::Pixel::statusKnown::part_known) {
                    should_split = true;
                }
                else if (variance_state == se::VarianceState::Gradient) {
                    should_split = true;
                }

                projects_inside = (pooling_pixel.status_known == se::Pixel::known);
            }
        }

        if (should_split) {
            if (oct_ptr->is_block) {
                allocation_list_local.block_list.push_back(oct_ptr);
                allocation_list_local.variance_state_list.push_back(variance_state);
                allocation_list_local.projects_inside_list.push_back(projects_inside);
            }
            else {
                NodeType* node_ptr = static_cast<NodeType*>(oct_ptr);
                const int child_size = node_ptr->size / 2;
                octree_.allocateChildren(node_ptr);
                for (int child_idx = 0; child_idx < 8; ++child_idx) {
                    stack.push_back(Task{node_ptr->getChildCoord(child_idx),
                                         child_size,
                                         oct_depth + 1,
                                         node_ptr->getChild(child_idx)});
                }
            }
        }
        else {
            assert(oct_depth);

            if (oct_ptr->is_block) {
                allocation_list_local.block_list.push_back(oct_ptr);
                allocation_list_local.variance_state_list.push_back(variance_state);
                allocation_list_local.projects_inside_list.push_back(projects_inside);
            }
            else if (variance_state == se::VarianceState::Constant) {
                allocation_list_local.node_list.push_back(oct_ptr);
            }
        }
    }
}

} // namespace se

#endif // SE_VOXEL_CARVER_IMPL_HPP
