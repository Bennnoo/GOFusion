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
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#ifndef INCLUDE_OKVIS_SUBMAPPINGINTERFACE_HPP
#define INCLUDE_OKVIS_SUBMAPPINGINTERFACE_HPP

#include <okvis/ThreadedSlam.hpp>
#include <okvis/DepthUtils.hpp>
#include <okvis/VoxelGridFilter.hpp>
#include <okvis/SubmappingUtils.hpp>
#include <okvis/mapTypedefs.hpp>
#include <Eigen/StdVector>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <se/common/math_util.hpp>
#include <unordered_map>
#include <unordered_set>
#include <torch/torch.h>
#include "gs/gaussian_utils.cuh"

namespace okvis {

  typedef se::Image<float> DepthFrame;
  using DepthMeasurementMap = std::map<size_t, std::vector<okvis::CameraMeasurement>>;
  using DepthMeasurementPtr = std::shared_ptr<DepthMeasurementMap>;
  typedef okvis::threadsafe::Queue<DepthMeasurementPtr> DepthFrameQueue;
  typedef okvis::threadsafe::Queue<LidarMeasurement, Eigen::aligned_allocator<LidarMeasurement>> LidarMeasurementQueue;
  typedef okvis::kinematics::Transformation Transformation;

    struct OkvisUpdate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        State latestState;
        okvis::Time timestamp;
        TrackingState trackingState;
        std::set<StateId> affectedStates;
        std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates;

        OkvisUpdate(const State &latestState = State(),
                    const okvis::Time &timestamp = okvis::Time(),
                    const TrackingState &trackingState = TrackingState(),
                    const std::set<StateId> affectedStates = std::set<StateId>(),
                    std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates = std::make_shared<const okvis::AlignedMap<StateId, State>>(okvis::AlignedMap<StateId, State>()))
                : latestState(latestState), timestamp(timestamp),
                  trackingState(trackingState), affectedStates(affectedStates), updatedStates(updatedStates) {};

        OkvisUpdate(const OkvisUpdate& newElem) = default;

        OkvisUpdate& operator=(const OkvisUpdate& newElem){
            latestState = newElem.latestState;
            timestamp = newElem.timestamp;
            trackingState = newElem.trackingState;
            affectedStates = newElem.affectedStates;
            updatedStates = newElem.updatedStates;
            return *this;
        }
    };

    typedef okvis::threadsafe::Queue<OkvisUpdate, Eigen::aligned_allocator<OkvisUpdate>> StateUpdatesQueue;

    struct SiFrameData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t id;
        Transformation T_WS;
        std::set<StateId> covisibles_;

        SiFrameData(const uint64_t &id = 0,
                    const Transformation &T_WS = Transformation::Identity(),
                    const std::set<StateId> &covisibles = std::set<StateId>())
                : id(id), T_WS(T_WS), covisibles_(covisibles) {};
    };

    typedef std::map<uint64_t, SiFrameData, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, SiFrameData>>> siFrameDataMap;

    typedef std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>, 
              Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>> RayVector;

    struct SupereightFrames {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t keyFrameId; 
        siFrameDataMap frameData_;
        bool loop_closure;
        RayVector vecRayMeasurements;
        std::vector<okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>>> vecDepthFrames;

        SupereightFrames(const uint64_t &keyframeId = 0,
                        const siFrameDataMap &FrameDataMap = siFrameDataMap{},
                        const bool loop_closure = false,
                        std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector3f>, Eigen::aligned_allocator<std::pair<Eigen::Matrix4f, Eigen::Vector3f>>>* vecRayMeasurementsVal = nullptr,
                        std::vector<okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Matrix4f, CameraMeasurement>>>>* vecDepthFramesVal = nullptr)
                          : keyFrameId(keyframeId), frameData_(FrameDataMap), 
                            loop_closure(loop_closure) {

                        if(vecRayMeasurementsVal){
                          for(size_t i = 0; i < vecRayMeasurementsVal->size(); i++){
                            vecRayMeasurements.push_back({Eigen::Isometry3f(vecRayMeasurementsVal->at(i).first),
                                                          vecRayMeasurementsVal->at(i).second});
                          }
                        }

                        if(vecDepthFramesVal){
                          for(size_t i = 0; i < vecDepthFramesVal->size(); i++){
                            okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>> measurement;
                            for(auto& elements : vecDepthFramesVal->at(i)){
                              auto& posed_idx_measurements = measurement[elements.first];
                              for(auto& element : elements.second) {
                                posed_idx_measurements.emplace_back(std::make_pair(Eigen::Isometry3f(element.first), element.second));
                              }
                            }
                            vecDepthFrames.push_back(measurement);
                          }
                        }

                  };
    };

    struct SubmapAlignmentTerm{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t frame_A_id; 
        uint64_t frame_B_id; 
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud_B; 
        std::vector<float> sigma_B; 
        VoxelHashMap voxelHashMap;

        SubmapAlignmentTerm() : frame_A_id(0), frame_B_id(0), voxelHashMap(0.06, 0.0, 1)
        {
        };

        SubmapAlignmentTerm(const uint64_t& frame_A_id, const uint64_t& frame_B_id)
          : frame_A_id(frame_A_id), frame_B_id(frame_B_id), voxelHashMap(0.06, 0.0, 1)
        {
        };

        SubmapAlignmentTerm(const uint64_t& frame_A_id, const uint64_t& frame_B_id, float voxel_grid_resolution)
          : frame_A_id(frame_A_id), frame_B_id(frame_B_id), voxelHashMap(voxel_grid_resolution, 0.0, 1)
        {
        };

    };

    typedef okvis::threadsafe::Queue<SupereightFrames, Eigen::aligned_allocator<SupereightFrames>> SupereightFrameQueue;

    typedef std::function<void(const std::map<uint64_t, Eigen::Vector3f>&)>
        gsOptimizedTrajectoryCallback;
    typedef std::function<void(const std::map<uint64_t, Eigen::Matrix4f>&)>
        gsOptimizedPoseCallback;

    struct GsPreviewFrame {
        int id = -1;
        int global_iter = -1;
        int num_splats = 0;
        int num_kf = 0;
        uint64_t num_submaps = 0;
        float input_fps = 0.0f;
        float tracking_fps = 0.0f;
        float fps = 0.0f;
        float backend_fps = 0.0f;
        cv::Mat rgb;
        cv::Mat depth;
        cv::Mat rendered_rgb;
    };

    struct OccupancyPreviewFrame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using PointVector = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
        using ColourVector = std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>;

        struct SubmapCloud {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            uint64_t submap_id = 0;
            std::shared_ptr<PointVector> occupied_points_local;
            std::shared_ptr<ColourVector> occupied_colours;
            std::shared_ptr<ColourVector> occupied_fisher_colours;
            std::shared_ptr<ColourVector> occupied_global_fisher_colours;
            Eigen::Matrix4f T_WK = Eigen::Matrix4f::Identity();
            size_t num_points = 0;
            uint64_t geometry_revision = 0;
            uint64_t colour_revision = 0;
            bool geometry_updated = false;
        };

        std::shared_ptr<std::vector<SubmapCloud>> submap_clouds;

        Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity();
        size_t num_submaps = 0;
        size_t num_points = 0;

        uint64_t integration_counter = 0;
    };

    struct OccupancyViewerConfig {
        bool enable = false;
        int update_interval = 5;
        int voxel_stride = 1;
        int max_points = 400000;

        int occupied_point_size = 2;

        bool show_pose = true;
        int max_fps = 15;
        std::string window_name = "GO Occupancy";
        int window_width = 960;
        int window_height = 600;

        int fisher_update_interval = 30;
        float fisher_display_regularizer = 0.1f;
        float fisher_eig_regularizer = 1.0e-6f;
    };


    struct GsDensifyConfig {
        bool enabled = false;
        int interval = 50;
        int tile_size = 16;
        float top_tiles_ratio = 0.02f;
        float top_peaks_ratio = 0.05f;
        int max_new_gaussians = 1024;
        float init_scale_mul = 1.0f;
        float init_z_scale = 5.0e-6f;
        float split_size_ratio = 1.5f;
        int split_factor = 2;
        float split_scale_shrink = 0.7f;
    };

    struct GsFinalizationSummary {
        bool valid = false;
        double pre_global_opt_system_fps = -1.0;
        double global_optimization_time_sec = 0.0;
        size_t global_opt_kf_count = 0;
        size_t save_iteration_frames = 0;
        size_t final_global_gaussian_count = 0;
        double gpu_memory_usage_mb = -1.0; 
        bool render_requested = false;
        size_t render_frame_count = 0;
        size_t render_saved_count = 0;
        size_t render_pose_miss_count = 0;
        std::string render_output_directory;
        bool joint_pose_applied = false;
        size_t joint_pose_updated_rows = 0;
    };


    class SubmappingInterface {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static constexpr size_t UNINITIALIZED_ID = std::numeric_limits<size_t>::max();

        SubmappingInterface() = delete;

        SubmappingInterface(
                const SupereightMapType::Config &mapConfig,
                const SupereightMapType::DataConfigType &dataConfig,
                const se::SubMapConfig &submapConfig,
                const okvis::ViParameters &parameters,
                std::shared_ptr<const std::map<uint64_t, Transformation>> gtPoses = nullptr)
                : mapConfig_(mapConfig), dataConfig_(dataConfig), submapConfig_(submapConfig),
                  meshesPath_(submapConfig.resultsDirectory),
                  viParameters_(parameters),
                  gtPoses_(std::move(gtPoses)),
                  sensorMeasurementDownsampling_(submapConfig.sensorMeasurementDownsampling),
                  depthQueueSize_(std::max<size_t>(submapConfig.depthQueueSize, 1)),
                  depthQueueDropIfFull_(submapConfig.depthQueueDropIfFull),
                  nonBlockingStateUpdateQueueSize_(
                      submapConfig.nonBlockingStateUpdateQueueSize > 0
                          ? submapConfig.nonBlockingStateUpdateQueueSize
                          : (submapConfig.realtimePipelineEnabled
                                 ? size_t(1)
                                 : size_t(100000))),
                  nonBlockingSupereightQueueSize_(
                      submapConfig.nonBlockingSupereightQueueSize > 0
                          ? submapConfig.nonBlockingSupereightQueueSize
                          : (submapConfig.realtimePipelineEnabled
                                 ? size_t(1)
                                 : size_t(5000))),
                  gsMappingEnabled_(submapConfig.enableGsMapping),
                  depthImageResDownsamplingRatio_(submapConfig.depthImageResDownsampling),
                  depthScaleFactor_(submapConfig.depthScalingFactor){

          blocking_ = false;

          bool camInMapping = false;

          for(size_t i = 0; i < parameters.nCameraSystem.numCameras(); i++){
            if(parameters.nCameraSystem.cameraType(i).isUsedMapping
               && (!parameters.mappingInputPreprocessor
                   || !parameters.mappingInputPreprocessor->enabled()
                   || parameters.mappingInputPreprocessor->targetCameraIndex() == i)) {
              LOG(INFO) << "Camera with index " << i << " will be used for mapping";
              camInMapping = true;
              const bool mapping_input_preprocessed =
                  parameters.mappingInputEnabledFor(i);
              const size_t geometry_factor = mapping_input_preprocessed
                                                 ? static_cast<size_t>(
                                                       submapConfig.depthImageResDownsampling)
                                                 : size_t(1);
              const auto camera =
                  parameters.mappingCameraGeometry(i, geometry_factor);
              se::PinholeCamera::Config camConfig;
              Eigen::VectorXd intrinsics;
              camera->getIntrinsics(intrinsics);

              camConfig.fx = intrinsics(0);
              camConfig.fy = intrinsics(1);
              camConfig.cx = intrinsics(2);
              camConfig.cy = intrinsics(3);
              const auto T_SC = parameters.mappingT_SC(i);
              camConfig.T_BS = T_SC->T().cast<float>();
              camConfig.width = camera->imageWidth();
              camConfig.height = camera->imageHeight();
              camConfig.near_plane = submapConfig.near_plane;
              camConfig.far_plane = submapConfig.far_plane;
              if (!mapping_input_preprocessed) {
                (void) validatedDepthDownsampledSize(
                    camConfig.width,
                    camConfig.height,
                    submapConfig.depthImageResDownsampling,
                    "mapping camera");
              }

              if(!cameraSensors_) {
                cameraSensors_ = okvis::AlignedMap<uint64_t,  std::pair<std::shared_ptr<const okvis::kinematics::Transformation>, se::PinholeCamera>>();
              }
              const se::PinholeCamera sensor = mapping_input_preprocessed
                  ? se::PinholeCamera(camConfig)
                  : se::PinholeCamera(
                        camConfig, submapConfig.depthImageResDownsampling);
              (*cameraSensors_).emplace(
                  i, std::make_pair(std::move(T_SC), std::move(sensor)));

            }

          }

          if(camInMapping && parameters.lidar) {
            throw std::runtime_error("There are LiDAR and cameras with mapping enabled in your config file, currently there is only support for either depth cameras or LiDAR at the same time");
          } else if(parameters.lidar) {
            submapAlignBlock_ = SubmapAlignmentTerm(0, 0, submapConfig.voxelGridResolution);

            se::Lidar::Config lidarConfig;
            lidarConfig.elevation_resolution_angle_ = (*parameters.lidar).elevation_resolution_angle;
            lidarConfig.azimuth_resolution_angle_ = (*parameters.lidar).azimuth_resolution_angle;
            lidarConfig.far_plane = submapConfig.far_plane;
            lidarConfig.near_plane = submapConfig.near_plane;
            lidarSensors_ = std::pair<okvis::kinematics::Transformation, se::Lidar>(std::make_pair((*parameters.lidar).T_SL,
                                                                                                         se::Lidar(lidarConfig)));
          }

          if (mapConfig_.res > 0.0f && mapConfig_.dim.maxCoeff() > 0.0f) {
            const float requested_voxels = mapConfig_.dim.maxCoeff() / mapConfig_.res;
            const unsigned int min_octree_size =
                static_cast<unsigned int>(2 * SupereightMapType::BlockType::size);
            const unsigned int effective_octree_size = se::math::power_two_up(
                std::max(requested_voxels, static_cast<float>(min_octree_size)));
            const float effective_map_dim =
                static_cast<float>(effective_octree_size) * mapConfig_.res;
            const float map_half_extent = 0.5f * effective_map_dim;
            if (submapConfig_.far_plane >= map_half_extent) {
              LOG(WARNING) << "Map half-extent (" << map_half_extent
                           << " m) <= far_plane (" << submapConfig_.far_plane
                           << " m). Distant occupancy will be truncated by map bounds. "
                              "Increase map.dim in se2.yaml or lower far_plane.";
            }
          }

          trajectoryLocked_ = false;

        };

        ~SubmappingInterface() { shutdown(); };

        bool addDepthMeasurement(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames);

        bool start();

        bool stateUpdateCallback(const State &latestState,
                                 const TrackingState &latestTrackingState,
                                 std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates);


        void setBlocking(bool blocking) {
          blocking_ = blocking;
        }

        void processSupereightFrames();

        bool checkForAvailableData();

        void frameUpdater(okvis::OkvisUpdate& propagatedStateUpdate);

        void saveSubmap(uint64_t kfId);

        void updateSEFrameData(siFrameDataMap& seFData, siFrameDataMap& newFData){
            for(auto it : newFData) seFData[it.first] = it.second;
            return;
        }


        bool finishedIntegrating();

        bool isFinished(){
            std::lock_guard<std::mutex> l(finishMutex_);
            return isFinished_;
        }

        void setFinished(){
            {
              std::lock_guard<std::mutex> l(finishMutex_);
              isFinished_ = true;
            }
            if (prevKeyframeId_ != UNINITIALIZED_ID) {
              enqueueSubmapFinalization(prevKeyframeId_);
            }
            waitForSubmapFinalization();

        }

        typedef std::function<
                bool(const SupereightMapType* submap_A_ptr,
                     const SupereightMapType* submap_B_ptr,
                     const uint64_t& frame_A_id,
                     const uint64_t& frame_B_id,
                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                     std::vector<float> sensorError)> AlignCallback;

        void setAlignCallback(const AlignCallback& alignCallback) {
          alignCallback_ = alignCallback;
        }

        void saveAllSubmapMeshes();

        bool publishGsPreview(GsPreviewFrame& previewFrame);

        bool publishOccupancyPreview(OccupancyPreviewFrame& previewFrame);
        bool buildOccupancyPreviewSnapshot(OccupancyPreviewFrame& previewFrame,
                                           bool refresh_geometry = false);
        void setGsFisherVisualizationEnabled(bool enabled);
        bool isGsFisherVisualizationEnabled() const
        {
          return gsFisherVisualizationEnabled_.load();
        }
        void setGsGlobalFisherVisualizationEnabled(bool enabled);
        bool isGsGlobalFisherVisualizationEnabled() const
        {
          return gsGlobalFisherVisualizationEnabled_.load();
        }


        void setGsDensifyConfig(const GsDensifyConfig& config);

        void setOccupancyViewerConfig(const OccupancyViewerConfig& config);
        void setGsOptimizedTrajectoryCallback(
            const gsOptimizedTrajectoryCallback& callback)
        {
          std::lock_guard<std::mutex> lock(gsOptimizedTrajectoryCallbackMtx_);
          gsOptimizedTrajectoryCallback_ = callback;
        }
        void setGsOptimizedPoseCallback(
            const gsOptimizedPoseCallback& callback)
        {
          std::lock_guard<std::mutex> lock(gsOptimizedPoseCallbackMtx_);
          gsOptimizedPoseCallback_ = callback;
        }

        bool finalizeGsMapping(const std::string& datasetPath,
                               const std::string& finalTrajectoryCsv,
                               const std::vector<ViSlamBackend::FinalTrajectoryPosition>&
                                   finalTrajectoryPoses =
                                       std::vector<ViSlamBackend::FinalTrajectoryPosition>());
        bool getLastGsFinalizationSummary(GsFinalizationSummary& summary) const {
          std::lock_guard<std::mutex> lock(gsFinalizationSummaryMtx_);
          if (!lastGsFinalizationSummary_.valid) {
            return false;
          }
          summary = lastGsFinalizationSummary_;
          return true;
        }

    private:
        AlignedUnorderedMap<uint64_t, se::Submap<SupereightMapType>> seSubmapLookup_;


        AlignedUnorderedMap<uint64_t, se::TriangleMesh<SupereightMapType::DataType::col_>> seMeshLookup_;

        struct SubmapOverlapCandidates {
          std::vector<uint64_t> covisible_older_candidates;
          std::vector<std::pair<float, uint64_t>> iou_ranked_candidates;
          std::vector<uint64_t> verified_ranked_ids;
          std::vector<uint64_t> verified_oldest_first_ids;
        };

        std::vector<uint64_t> collectCovisibleOlderSubmaps(
            uint64_t activeId, uint64_t previousId, int recursionDepth = 15);

        SubmapOverlapCandidates collectOverlappingSubmaps(
            uint64_t activeId,
            uint64_t previousId,
            const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
                queryPointsB);

        uint64_t findOldestOverlappingSubmap(std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downsampledPointCloud,
                                           uint64_t activeId, uint64_t previousId);

        size_t determineObservedPoints(uint64_t& mapIdA, uint64_t& mapIdB,
                                       const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints) const;

std::vector<std::pair<float, uint64_t>> rankSubmapCandidatesByIou(
            uint64_t activeId,
            const std::vector<uint64_t>& candidateIds,
            size_t topK = 5,
            float minIou = 0.2f) const;
std::vector<uint64_t> verifySubmapCandidatesByObservedPoints(
            uint64_t activeId,
            const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
                queryPointsB,
            const std::vector<std::pair<float, uint64_t>>& rankedCandidates,
            size_t minObservedPoints) const;
        size_t determineObservedPointsAndSigma(uint64_t& mapIdA, uint64_t& mapIdB,
                                               const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                               std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints,
                                               const std::vector<float>& sigma, std::vector<float>& observedSigma) const;

        float evaluateDepthOverlap(const SupereightMapType& map, uint64_t& mapId, SupereightFrames& seFrame);

        void integrationLoop();

        bool predict(const okvis::Time &finalTimestamp,
                     const Transformation &T_SC,
                     Transformation &T_WC,
                     StateId* sourceStateId = nullptr);

        DepthFrame depthMat2ImageView(const cv::Mat &inputDepth);

        se::Image<se::colour_t> rgbMat2ImageView(const cv::Mat &inputRGB);


        void obtainCovisibles(std::vector<uint64_t>& orderedIdx, const okvis::StateId& latestCovisible, int recursionDepth);

        bool decideNewSubmap(SupereightFrames& supereightFrame);

        void addSubmapAlignmentFactors(const SupereightFrames& supereightFrame);

        void depthMap2MapFactors();

        void updateDepthAlignBlock(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
          const kinematics::Transformation& T_WK, const int& depthImage_idx);

        struct OccupancyBlockKey {
          int x = 0;
          int y = 0;
          int z = 0;

          bool operator==(const OccupancyBlockKey& other) const
          {
            return x == other.x && y == other.y && z == other.z;
          }
        };

        struct OccupancyBlockKeyHash {
          std::size_t operator()(const OccupancyBlockKey& key) const
          {
            std::size_t seed = std::hash<int>{}(key.x);
            seed ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
          }
        };

        struct OccupancyTrackedBlock {
          OccupancyBlockKey key;
          OccupancyPreviewFrame::PointVector occupied_points_local;
          OccupancyPreviewFrame::ColourVector occupied_colours;
          bool active = true;
        };

        struct OccupancySubmapCache {
          std::shared_ptr<OccupancyPreviewFrame::PointVector> occupied_points_local;
          std::shared_ptr<OccupancyPreviewFrame::ColourVector> occupied_colours;
          std::shared_ptr<OccupancyPreviewFrame::ColourVector> occupied_fisher_colours;
          std::shared_ptr<OccupancyPreviewFrame::ColourVector> occupied_global_fisher_colours;
          std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash> fisher_score_grid;
          float fisher_cell_size = 0.0f;
          std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash>
              global_fisher_score_grid;
          float global_fisher_cell_size = 0.0f;
          size_t num_points = 0;
          uint64_t geometry_revision = 0;
          uint64_t colour_revision = 0;
          int cached_stride = -1;
          int cached_occ_budget = -1;
          std::vector<OccupancyTrackedBlock> tracked_blocks;
          std::unordered_map<OccupancyBlockKey, size_t, OccupancyBlockKeyHash> tracked_block_indices;
          std::deque<OccupancyBlockKey> dirty_block_queue;
          std::unordered_set<OccupancyBlockKey, OccupancyBlockKeyHash> dirty_block_set;
          bool needs_full_rebuild = true;
        };

        void markOccupancySubmapDirty(
            uint64_t submap_id,
            const std::unordered_set<const se::OctantBase*>* updated_octants = nullptr);
        bool rebuildOccupancySubmapCache(uint64_t submap_id,
                                         const se::Submap<SupereightMapType>& submap,
                                         int stride,
                                         int occ_budget);


        void updateGsFisherVisualization();
        struct GsRuntime;
        struct GsFisherDiag {
          torch::Tensor xyz;
          torch::Tensor opacity;
        };

        struct GsFisherScoreConfig {
          float weight_xyz = 1.0f;
          float weight_opacity = 1.0f;
        };

        GsFisherDiag regularizeInvertGsFisherDiag(const GsFisherDiag& fisher_diag,
                                                 float lambda) const;
        GsFisherDiag addGsFisherDiag(const GsFisherDiag& lhs, const GsFisherDiag& rhs) const;

        void invalidateGsSubmapFisherCache(uint64_t submap_id);
        bool ensureGsSubmapModelLoaded(uint64_t submap_id);
        torch::Tensor gsSubmapXyzCpu(uint64_t submap_id);
        void updateGsSubmapFisherFromTrainingSamples(
            uint64_t submap_id,
            const gs::FisherTrainingSampleVector& fisher_training_samples);

        torch::Tensor collapseGsFisherDiagPerGaussian(const GsFisherDiag& fisher_diag,
                                                      const GsFisherScoreConfig& score_config) const;
        torch::Tensor normalizePerGaussianScoreForDisplay(const torch::Tensor& per_gaussian_score) const;
        bool updateGsActiveFisherVisualization();
        bool updateGsGlobalFisherVisualization();
        void clearInactiveFisherOverlays(uint64_t active_submap_id);
        void clearGlobalFisherOverlay();
        void normalizeGlobalFisherScoreGridForDisplay();
        void rebuildGlobalFisherOverlay();
        void recolorOccupancySubmapFromGlobalFisher(
            const uint64_t submap_id, OccupancySubmapCache& cache) const;
        bool projectAllGsFisherToGlobalOccupancy();
        void sampleOccupancyBlockFromPointer(const se::Submap<SupereightMapType>& submap,
                                             const SupereightMapType::BlockType* block_ptr,
                                             int stride,
                                             OccupancyTrackedBlock& tracked_block) const;
        bool sampleOccupancyBlock(const se::Submap<SupereightMapType>& submap,
                                  const OccupancyBlockKey& block_key,
                                  int stride,
                                  OccupancyTrackedBlock& tracked_block) const;
        void recolorOccupancySubmapFromFisher(OccupancySubmapCache& cache) const;
        void rebuildOccupancySubmapPoints(uint64_t submap_id,
                                          OccupancySubmapCache& cache,
                                          int occ_budget);

        void updateOccupancyPreview(const State& current_state);
        void enqueueSubmapFinalization(uint64_t submap_id);
        void runSubmapFinalizationWorker();
        void waitForSubmapFinalization();
        void updateGsSubmapResidencyPolicy(uint64_t active_submap_id);
        bool clearFrozenGsTrainingResources(uint64_t submap_id);
        bool enqueueGsModelOffload(uint64_t submap_id);

        void runGsPreviewWorker(std::shared_ptr<GsRuntime> runtime);
        void stopGsPreviewWorker();

        void waitForGsModelOffloads();
        void shutdown();
        void trimGsCudaCache();

std::unordered_set<const se::OctantBase*> updated_octants_;
Eigen::Matrix4f currentPose_ = Eigen::Matrix4f::Identity();
std::mutex currentPose_mtx_;
        SupereightMapType::Config mapConfig_;       
        SupereightMapType::DataConfigType dataConfig_; 
        se::SubMapConfig submapConfig_; 
        std::string meshesPath_;  

        DepthFrameQueue
                depthMeasurements_; 
        LidarMeasurementQueue
                lidarMeasurements_; 
        StateUpdatesQueue
                stateUpdates_; 
        SupereightFrameQueue
                supereightFrames_; 

        std::mutex finishMutex_; 

        gsOptimizedTrajectoryCallback gsOptimizedTrajectoryCallback_;
        std::mutex gsOptimizedTrajectoryCallbackMtx_;
        gsOptimizedPoseCallback gsOptimizedPoseCallback_;
        std::mutex gsOptimizedPoseCallbackMtx_;

        bool blocking_;

        struct LatestKeyframe {
            uint64_t Id;
            bool started;

            LatestKeyframe() : Id(1), started(false) {};
        } latestKeyframe;


        okvis::Trajectory propagatedStates;
        std::atomic_bool trajectoryLocked_; 

        AlignedMap<StateId, std::set<StateId>> covisibleFrames_;

        uint32_t integration_counter_ = 0;
        uint32_t finishedIntegratingSensorOnlyCycles_ = 0;

        bool mappingFpsTrackingStartSet_ = false;
        okvis::Time mappingFpsTrackingStartTime_ = okvis::Time(0);
        std::mutex mappingFpsMtx_;

        std::thread submapIntegration_;

        std::thread dataIntegration_;

        uint64_t prevKeyframeId_ = UNINITIALIZED_ID;

        uint64_t previousSubmapId_ = UNINITIALIZED_ID;

        uint64_t curKeyframeId_ = 0;

        unsigned int keyframeCounter_ = 0;

        unsigned int submapCounter_ = 0;
        std::atomic<uint64_t> gsPreviewSubmapCount_{0};

        std::unique_ptr<SupereightFrames> frame_;

        bool isFinished_ = false;
        std::atomic_bool isProcessingSeFrame_{false};

        std::optional<okvis::AlignedMap<uint64_t,  std::pair<std::shared_ptr<const okvis::kinematics::Transformation>, se::PinholeCamera>>> cameraSensors_; 
        std::optional<std::pair<Transformation, se::Lidar>> lidarSensors_; 

        const okvis::ViParameters viParameters_;
        const std::shared_ptr<const std::map<uint64_t, Transformation>> gtPoses_;

        const int sensorMeasurementDownsampling_;

        int sensorMeasurementDownsamplingCounter_ = 0;

        const size_t depthQueueSize_;
        const bool depthQueueDropIfFull_;
        const size_t nonBlockingStateUpdateQueueSize_;
        const size_t nonBlockingSupereightQueueSize_;

        std::atomic<uint64_t> realtimeDepthQueueDropCount_{0};
        std::atomic<uint64_t> realtimeDepthDrainDropCount_{0};
        std::atomic<uint64_t> realtimeStateQueueDropCount_{0};
        std::atomic<uint64_t> realtimeSupereightQueueDropCount_{0};
        std::atomic<uint64_t> realtimeSupereightDrainDropCount_{0};
        std::atomic<uint64_t> realtimePackedDepthDropCount_{0};

        const int depthImageResDownsamplingRatio_;

        float depthScaleFactor_ = 1.0f;


        struct DepthOverlapSamplingCacheKey {
          int camIdx = -1;
          int width = 0;
          int height = 0;
          int skipInterval = 0;

          bool operator==(const DepthOverlapSamplingCacheKey& other) const
          {
            return camIdx == other.camIdx && width == other.width && height == other.height
                   && skipInterval == other.skipInterval;
          }
        };

        struct DepthOverlapSamplingCacheKeyHash {
          std::size_t operator()(const DepthOverlapSamplingCacheKey& key) const
          {
            std::size_t seed = std::hash<int>{}(key.camIdx);
            seed ^= std::hash<int>{}(key.width) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(key.height) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(key.skipInterval) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
          }
        };

        struct DepthOverlapSamplingCache {
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
          Eigen::Matrix3Xf sampleDirections;
          std::vector<int> pixelIndices;
          std::vector<int> validSampleIndices;
        };

        std::unordered_map<DepthOverlapSamplingCacheKey,
                           DepthOverlapSamplingCache,
                           DepthOverlapSamplingCacheKeyHash>
            depthOverlapSamplingCache_;

        std::atomic_bool lidarKfDetected_ = false;
        unsigned int numIntegratedLidarFrames_ = 0;
        unsigned int numIntegratedDepthFrames_ = 0;
        SupereightMapType* previousSubmap_ = nullptr; 

        AlignCallback alignCallback_;


        State latestState_;
        std::mutex state_mtx_;

        SubmapAlignmentTerm submapAlignBlock_;

        threadsafe::Queue<GsPreviewFrame> gsPreviewQueue_;
        threadsafe::Queue<OccupancyPreviewFrame> occupancyPreviewQueue_;
        struct SubmapFinalizeTask {
          uint64_t submap_id = UNINITIALIZED_ID;
          std::shared_ptr<SupereightMapType> map;
        };
        std::thread submapFinalizeWorker_;
        std::mutex submapFinalizeMutex_;
        std::condition_variable submapFinalizeCv_;
        std::deque<SubmapFinalizeTask> submapFinalizeQueue_;
        std::unordered_set<uint64_t> submapFinalizePending_;
        size_t submapFinalizeInFlight_ = 0;
        bool submapFinalizeStop_ = false;
        std::mutex submapMeshMutex_;

        struct GsModelOffloadTask {
          uint64_t submap_id = UNINITIALIZED_ID;
          std::shared_ptr<GsRuntime> runtime;
          std::unique_ptr<gs::GaussianModel> model;

        };
        std::shared_ptr<GsRuntime> gsRuntime_;
        std::shared_ptr<GsRuntime> gsPreviewRuntime_;
        std::thread gsPreviewWorker_;

        std::mutex gsModelOffloadMutex_;
        std::deque<GsModelOffloadTask> gsModelOffloadQueue_;
        bool gsModelOffloadStop_ = false;
        size_t gsIntegrationFailureCount_ = 0;
        bool gsRuntimeDisabledDueToErrors_ = false;

        std::map<uint64_t, uint64_t> gsPoseStateIdByTimestamp_;
        std::mutex gsPoseStateIdByTimestampMtx_;
        std::mutex shutdownMutex_;
        bool shutdownComplete_ = false;
        GsDensifyConfig gsDensifyConfig_;
        std::atomic_bool gsMappingEnabled_ = false;
        std::atomic_bool gsFisherVisualizationEnabled_{false};
        std::atomic_bool gsGlobalFisherVisualizationEnabled_{false};

        std::atomic_bool gsFisherVisualizationRefreshRequested_{false};
        std::atomic_bool gsGlobalFisherVisualizationRefreshRequested_{false};
        OccupancyViewerConfig occupancyViewerConfig_;
        GsFisherScoreConfig gsFisherScoreConfig_;
        std::unordered_map<uint64_t, OccupancySubmapCache> occupancySubmapCaches_;
        std::unordered_set<uint64_t> occupancyDirtySubmaps_;

        std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash> globalFisherScoreGrid_;
        std::unordered_map<OccupancyBlockKey, float, OccupancyBlockKeyHash>
            globalFisherDisplayGrid_;
        float globalFisherCellSize_ = 0.0f;
        uint64_t gsFisherLastUpdateCounter_ = 0;
        uint64_t gsGlobalFisherLastUpdateCounter_ = 0;
        mutable std::mutex gsFinalizationSummaryMtx_;
        GsFinalizationSummary lastGsFinalizationSummary_;
        double gsGpuUsedMemoryStartMb_ = -1.0;

    };
} 
#endif 
