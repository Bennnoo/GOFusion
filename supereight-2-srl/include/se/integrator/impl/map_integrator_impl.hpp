/*
 * SPDX-FileCopyrightText: 2021-2024 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2021 Nils Funk
 * SPDX-FileCopyrightText: 2021-2024 Sotiris Papatheodorou
 * SPDX-FileCopyrightText: 2022-2024 Simon Boche
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#ifndef SE_MAP_INTEGRATOR_IMPL_HPP
#define SE_MAP_INTEGRATOR_IMPL_HPP

namespace se {
namespace details {

/**
 * Integration helper struct for partial function specialisation
 */
template<se::Field FldT, se::Res ResT>
struct IntegrateDepthImplD {
    template<typename SensorT, typename MapT>
    static void integrate(MapT& map,
                          const timestamp_t timestamp,
                          const Measurements<SensorT>& measurements,
                          std::unordered_set<const OctantBase*>* const updated_octants);
};

#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
/**
 * Integration helper struct for partial function specialisation (GS path).
 */
template<se::Field FldT, se::Res ResT>
struct GSIntegrateDepthImplD {
    template<typename SensorT, typename MapT>
    static void integrate(MapT& map,
                          const timestamp_t timestamp,
                          const Measurements<SensorT>& measurements,
                          gs::GaussianModel& gs_model,
                          std::vector<gs::Camera>& gs_cam_list,
                          std::vector<torch::Tensor>& gt_img_list,
                          std::vector<gs::GeometryTarget>& geometry_target_list,
                          std::vector<torch::Tensor>& depth_target_list_cpu,
                          gs::DataQueue& data_queue,
                          std::unordered_set<const OctantBase*>* const updated_octants,
                          std::vector<gs::Camera>* gs_cam_list_cpu = nullptr,
                          std::vector<torch::Tensor>* gt_img_list_cpu = nullptr,
                          gs::PoseUpdateVector* pose_updates = nullptr,
                          gs::FisherTrainingSampleVector* fisher_training_samples = nullptr,
                          gs::DataPacket* preview_packet = nullptr,
                          bool* occupancy_completed = nullptr);
};
#endif

/**
 * Integration helper struct for partial function specialisation
 */
template<se::Field FldT, se::Res ResT>
struct IntegrateRayBatchImplD {
    template<typename SensorT, typename MapT>
    static void integrate(
        MapT& map,
        const SensorT& sensor,
        const std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                          Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>>&
            rayPoseBatch,
        const timestamp_t timestamp,
        std::unordered_set<const OctantBase*>* const updated_octants);
};

/**
 * TSDF integration helper struct for partial function specialisation
 */
template<Res ResT>
struct IntegrateDepthImplD<se::Field::TSDF, ResT> {
    template<typename SensorT, typename MapT>
    static void integrate(MapT& map,
                          const timestamp_t timestamp,
                          const Measurements<SensorT>& measurements,
                          std::unordered_set<const OctantBase*>* const updated_octants)
    {
        assert(measurements.depth.sensor.model.imageWidth() == measurements.depth.image.width());
        assert(measurements.depth.sensor.model.imageHeight() == measurements.depth.image.height());
        // Allocation
        TICK("allocation")
        se::RaycastCarver raycast_carver(map,
                                         measurements.depth.sensor,
                                         measurements.depth.image,
                                         measurements.depth.T_WC,
                                         timestamp);
        std::vector<OctantBase*> block_ptrs = raycast_carver();
        TOCK("allocation")

        // Update
        TICK("update")
        se::Updater updater(map, block_ptrs, timestamp, measurements);
        TOCK("update")

        if (updated_octants) {
            // TODO: Currently returning allocated, not updated octants. Remove non-updated blocks from
            // block_ptrs during the call to se::Updater::update() to return only updated octants.
            updated_octants->insert(block_ptrs.begin(), block_ptrs.end());
        }
    }
};

/**
 * Multi-res OFusion integration helper struct for partial function specialisation
 */
template<>
struct IntegrateDepthImplD<se::Field::Occupancy, se::Res::Multi> {
    template<typename SensorT, typename MapT>
    static void integrate(MapT& map,
                          const timestamp_t timestamp,
                          const Measurements<SensorT>& measurements_,
                          std::unordered_set<const OctantBase*>* const updated_octants)
    {
        Measurements<SensorT> measurements = measurements_;
        assert(measurements.depth.image.width() == measurements.depth.sensor.model.imageWidth());
        assert(measurements.depth.image.height() == measurements.depth.sensor.model.imageHeight());
        // Allocation
        TICK("allocation")
        VolumeCarver<MapT, SensorT> volume_carver(
            map,
            measurements.depth.sensor,
            measurements.depth.image,
            measurements.depth.T_WC,
            timestamp); //< process based on variance state and project inside
        se::VolumeCarverAllocation allocation_list = volume_carver();
        TOCK("allocation")

        // Update
        TICK("update")
        se::Updater updater(map, timestamp, measurements);
        updater(allocation_list, updated_octants);
        TOCK("update")
    }
};

#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
/**
 * Multi-res OFusion + GS integration helper struct for partial function specialisation.
 */
template<>
struct GSIntegrateDepthImplD<se::Field::Occupancy, se::Res::Multi> {
    template<typename SensorT, typename MapT>
    static void integrate(MapT& map,
                          const timestamp_t timestamp,
                          const Measurements<SensorT>& measurements_,
                          gs::GaussianModel& gs_model,
                          std::vector<gs::Camera>& gs_cam_list,
                          std::vector<torch::Tensor>& gt_img_list,
                          std::vector<gs::GeometryTarget>& geometry_target_list,
                          std::vector<torch::Tensor>& depth_target_list_cpu,
                          gs::DataQueue& data_queue,
                          std::unordered_set<const OctantBase*>* const updated_octants,
                          std::vector<gs::Camera>* gs_cam_list_cpu = nullptr,
                          std::vector<torch::Tensor>* gt_img_list_cpu = nullptr,
                          gs::PoseUpdateVector* pose_updates = nullptr,
                          gs::FisherTrainingSampleVector* fisher_training_samples = nullptr,
                          gs::DataPacket* preview_packet = nullptr,
                          bool* occupancy_completed = nullptr)
    {
        // A training sample is valid only when every indexed view was
        // appended. GSUpdater constructs the complete image/camera sample in
        // local temporaries before committing it to these persistent arrays.
        if ((gs_cam_list_cpu == nullptr) != (gt_img_list_cpu == nullptr)
            || gs_cam_list.size() != gt_img_list.size()
            || (gs_model.optimParams.geometrySupervisionActive()
                && (geometry_target_list.size() != gt_img_list.size()
                    || depth_target_list_cpu.size()
                        != (gt_img_list_cpu ? gt_img_list_cpu->size() : 0)))
            || (gs_cam_list_cpu
                && gs_cam_list_cpu->size() != gt_img_list_cpu->size())) {
            throw std::runtime_error(
                "GS training sample size mismatch before integration");
        }

        // Keep the occupancy path behaviour identical to the default integrator.
        Measurements<SensorT> measurements = measurements_;
        assert(measurements.depth.image.width() == measurements.depth.sensor.model.imageWidth());
        assert(measurements.depth.image.height() == measurements.depth.sensor.model.imageHeight());

        // Allocation
        TICK("allocation")
        VolumeCarver<MapT, SensorT> volume_carver(map,
                                                  measurements.depth.sensor,
                                                  measurements.depth.image,
                                                  measurements.depth.T_WC,
                                                  timestamp);
        se::VolumeCarverAllocation allocation_list = volume_carver();
        TOCK("allocation")

        // Update (occupancy + GS)
        TICK("update")
        if (occupancy_completed) {
            *occupancy_completed = false;
        }
        std::vector<const OctantBase*> updated_octants_vec;
        try {
            se::GSUpdater<MapT, SensorT> updater(map,
                                                 timestamp,
                                                 measurements,
                                                 gs_model,
                                                 gs_cam_list,
                                                 gt_img_list,
                                                 geometry_target_list,
                                                 depth_target_list_cpu,
                                                 data_queue,
                                                 gs_cam_list_cpu,
                                                 gt_img_list_cpu,
                                                 pose_updates,
                                                 fisher_training_samples,
                                                 preview_packet,
                                                 occupancy_completed
);
            updater(allocation_list, updated_octants ? &updated_octants_vec : nullptr);
        }
        catch (...) {
            if (updated_octants) {
                updated_octants->insert(updated_octants_vec.begin(),
                                        updated_octants_vec.end());
            }
            throw;
        }
        TOCK("update")

        if (updated_octants) {
            updated_octants->insert(updated_octants_vec.begin(), updated_octants_vec.end());
        }
    }
};
#endif

/**
 * Multi-res OFusion integration helper struct for partial function specialisation
 */
template<>
struct IntegrateRayBatchImplD<se::Field::Occupancy, se::Res::Multi> {
    template<typename SensorT, typename MapT>
    static void integrate(
        MapT& map,
        const SensorT& sensor,
        const std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                          Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>>&
            rayPoseBatch,
        const timestamp_t timestamp,
        std::unordered_set<const OctantBase*>* const updated_octants)
    {
        se::RayIntegrator<MapT, SensorT> rayIntegrator(
            map, sensor, rayPoseBatch[0].second, rayPoseBatch[0].first, timestamp, updated_octants);

        // do downsampling
        for (size_t i = 0; i < rayPoseBatch.size(); i++) {
            TICK("Ray Integration")
            TICK("allocation-integration")
            if (rayIntegrator.resetIntegrator(
                    rayPoseBatch[i].second, rayPoseBatch[i].first, timestamp)) {
                rayIntegrator();
            }
            TOCK("allocation-integration")
            TOCK("Ray Integration")
        }
        // Do Propagations
        TICK("propagateBlocksToCoarsestScale")
        rayIntegrator.propagateBlocksToCoarsestScale();
        TOCK("propagateBlocksToCoarsestScale")
        TICK("propagateToRoot")
        rayIntegrator.propagateToRoot();
        TOCK("propagateToRoot")
    }
};


template<typename MapT>
using IntegrateDepthImpl = IntegrateDepthImplD<MapT::fld_, MapT::res_>;

#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
template<typename MapT>
using GSIntegrateDepthImpl = GSIntegrateDepthImplD<MapT::fld_, MapT::res_>;
#endif

template<typename MapT>
using IntegrateRayBatchImpl = IntegrateRayBatchImplD<MapT::fld_, MapT::res_>;

} // namespace details

template<typename MapT>
MapIntegrator<MapT>::MapIntegrator(MapT& map) : map_(map)
{
}

template<typename MapT>
template<typename SensorT>
void MapIntegrator<MapT>::integrateDepth(
    const timestamp_t timestamp,
    const Measurements<SensorT>& measurements,
    std::unordered_set<const OctantBase*>* const updated_octants)
{
    se::details::IntegrateDepthImpl<MapT>::template integrate<SensorT>(
        map_, timestamp, measurements, updated_octants);
}

#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
template<typename MapT>
template<typename SensorT>
void MapIntegrator<MapT>::integrateDepth(
    const timestamp_t timestamp,
    const Measurements<SensorT>& measurements,
    gs::GaussianModel& gs_model,
    std::vector<gs::Camera>& gs_cam_list,
    std::vector<torch::Tensor>& gt_img_list,
    std::vector<gs::GeometryTarget>& geometry_target_list,
    std::vector<torch::Tensor>& depth_target_list_cpu,
    gs::DataQueue& data_queue,
    std::unordered_set<const OctantBase*>* const updated_octants,
    std::vector<gs::Camera>* gs_cam_list_cpu,
    std::vector<torch::Tensor>* gt_img_list_cpu,
    gs::PoseUpdateVector* pose_updates,
    gs::FisherTrainingSampleVector* fisher_training_samples,
    gs::DataPacket* preview_packet,
    bool* occupancy_completed)
{
    se::details::GSIntegrateDepthImpl<MapT>::template integrate<SensorT>(map_,
                                                                          timestamp,
                                                                          measurements,
                                                                          gs_model,
                                                                          gs_cam_list,
                                                                          gt_img_list,
                                                                          geometry_target_list,
                                                                          depth_target_list_cpu,
                                                                          data_queue,
                                                                          updated_octants,
                                                                          gs_cam_list_cpu,
                                                                          gt_img_list_cpu,
                                                                          pose_updates,
                                                                          fisher_training_samples,
                                                                          preview_packet,
                                                                          occupancy_completed
);
}
#endif

template<typename MapT>
template<typename SensorT>
void MapIntegrator<MapT>::integrateRayBatch(
    const timestamp_t timestamp,
    const std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                      Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>>&
        rayPoseBatch,
    const SensorT& sensor,
    std::unordered_set<const OctantBase*>* const updated_octants)
{
    se::details::IntegrateRayBatchImpl<MapT>::integrate(
        map_, sensor, rayPoseBatch, timestamp, updated_octants);
}

} // namespace se

#endif // SE_MAP_INTEGRATOR_IMPL_HPP
