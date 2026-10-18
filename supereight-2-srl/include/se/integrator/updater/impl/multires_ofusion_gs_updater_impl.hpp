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

#ifndef SE_MULTIRES_OFUSION_GS_UPDATER_IMPL_HPP
#define SE_MULTIRES_OFUSION_GS_UPDATER_IMPL_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <glog/logging.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "gs/error_driven_densify.cuh"
#include "gs/loss_utils.cuh"
#include "gs/monogs_render_utils.cuh"
#include "gs/nvtx_utils.hpp"
#include "gs/quad_tree.cuh"
#include "gs/render_utils.cuh"

#include "gs/gaussian_init_utils.hpp"

namespace {

static thread_local std::vector<gs::Leaf> g_qtree_leaves;
static thread_local std::vector<int32_t> g_qtree_leaf_id;
static thread_local int g_qtree_leaf_W = 0;
static thread_local int g_qtree_leaf_H = 0;
static thread_local int g_qtree_leaf_count = 0;

static std::atomic<bool> g_logged_empty_gs_model_skip{false};

struct GaussianCenterCandidate {
    gs::Point pos_w{};
    float depth_m = 0.f;
    int u_colour = 0;
    int v_colour = 0;
    int u_depth = 0;
    int v_depth = 0;
    float leaf_w_depth_px = 0.f;
    float leaf_h_depth_px = 0.f;
};

struct GaussianCandidateScratch {
    std::vector<GaussianCenterCandidate> candidates;
    std::vector<uint8_t> accepted;
    std::vector<int32_t> accepted_indices;
    std::vector<size_t> thread_counts;
    std::vector<size_t> thread_offsets;
    std::vector<gs::Point> positions;
    std::vector<gs::Color> colors;
    std::vector<gs::Point> scales_xyz;
    std::vector<gs::Quat> rotations_wxyz;
};

static thread_local GaussianCandidateScratch g_gaussian_candidate_scratch;

static inline int env_int_clamped(const char* name,
                                  const int fallback,
                                  const int minimum,
                                  const int maximum)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return std::clamp(static_cast<int>(parsed), minimum, maximum);
}

static inline bool env_bool(const char* name, const bool fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0
           && std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0;
}

static inline int adaptive_parallel_threads(const size_t item_count,
                                            const size_t min_items_per_thread = 512)
{
#ifdef _OPENMP
    const int max_threads = std::max(1, omp_get_max_threads());
#else
    const int max_threads = 1;
#endif
    if (max_threads == 1 || item_count < 2 * min_items_per_thread) {
        return 1;
    }
    const size_t useful = (item_count + min_items_per_thread - 1) / min_items_per_thread;
    return std::max(1, std::min(max_threads, static_cast<int>(useful)));
}

static inline size_t stable_compact_indices(const std::vector<uint8_t>& flags,
                                            std::vector<int32_t>& indices,
                                            std::vector<size_t>& counts,
                                            std::vector<size_t>& offsets,
                                            const int requested_threads)
{
    const size_t item_count = flags.size();
    const int thread_count = std::max(1, requested_threads);

    if (thread_count == 1) {
        indices.clear();
        for (size_t i = 0; i < item_count; ++i) {
            if (flags[i] != 0) {
                indices.push_back(static_cast<int32_t>(i));
            }
        }
        return indices.size();
    }

    counts.resize(static_cast<size_t>(thread_count));
    offsets.resize(static_cast<size_t>(thread_count));

    indices.resize(item_count);
    size_t total = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        const size_t begin = item_count * static_cast<size_t>(tid)
                             / static_cast<size_t>(thread_count);
        const size_t end = item_count * static_cast<size_t>(tid + 1)
                           / static_cast<size_t>(thread_count);
        size_t local_count = 0;
        for (size_t i = begin; i < end; ++i) {
            local_count += flags[i] != 0;
        }
        counts[static_cast<size_t>(tid)] = local_count;

#pragma omp barrier
#pragma omp single
        {
            for (int worker = 0; worker < thread_count; ++worker) {
                offsets[static_cast<size_t>(worker)] = total;
                total += counts[static_cast<size_t>(worker)];
            }
        }
#pragma omp barrier
        size_t output = offsets[static_cast<size_t>(tid)];
        for (size_t i = begin; i < end; ++i) {
            if (flags[i] != 0) {
                indices[output++] = static_cast<int32_t>(i);
            }
        }
    }
#else

    indices.clear();
    for (size_t i = 0; i < item_count; ++i) {
        if (flags[i] != 0) {
            indices.push_back(static_cast<int32_t>(i));
        }
    }
    return indices.size();
#endif
    indices.resize(total);
    return total;
}

static_assert(std::is_standard_layout_v<se::RGB>, "se::RGB must remain standard layout.");
static_assert(std::is_trivially_copyable_v<se::RGB>,
              "se::RGB must remain trivially copyable for zero-copy image views.");
static_assert(sizeof(se::RGB) == 3, "se::RGB must remain tightly packed RGB888.");

struct ProjectionTensorCache {
    int width = -1;
    int height = -1;
    float fx = 0.f;
    float fy = 0.f;
    float cx = 0.f;
    float cy = 0.f;
    int cuda_device_index = -1;
    torch::Tensor cpu;
    torch::Tensor cuda;
};

static thread_local ProjectionTensorCache g_projection_tensor_cache;

static inline bool projectionCacheMatches(const ProjectionTensorCache& cache,
                                          const int width,
                                          const int height,
                                          const float fx,
                                          const float fy,
                                          const float cx,
                                          const float cy)
{
    return cache.cpu.defined() && cache.width == width && cache.height == height && cache.fx == fx
           && cache.fy == fy && cache.cx == cx && cache.cy == cy;
}

static inline const torch::Tensor& getCachedProjectionMatrixCpu(const int width,
                                                                const int height,
                                                                const float fx,
                                                                const float fy,
                                                                const float cx,
                                                                const float cy)
{
    auto& cache = g_projection_tensor_cache;
    if (!projectionCacheMatches(cache, width, height, fx, fy, cx, cy)) {
        cache.width = width;
        cache.height = height;
        cache.fx = fx;
        cache.fy = fy;
        cache.cx = cx;
        cache.cy = cy;
        cache.cuda_device_index = -1;
        cache.cpu = gs::getProjectionMatrix(width, height, fx, fy, cx, cy).to(torch::kCPU).contiguous();
        cache.cuda = torch::Tensor();
    }
    return cache.cpu;
}

static inline const torch::Tensor& getCachedProjectionMatrixCuda(const int width,
                                                                 const int height,
                                                                 const float fx,
                                                                 const float fy,
                                                                 const float cx,
                                                                 const float cy,
                                                                 const torch::Device& device)
{
    auto& cache = g_projection_tensor_cache;
    const auto& proj_cpu = getCachedProjectionMatrixCpu(width, height, fx, fy, cx, cy);
    const int device_index = (device.has_index() ? device.index() : 0);
    if (!cache.cuda.defined() || cache.cuda_device_index != device_index
        || cache.cuda.device() != device) {
        cache.cuda_device_index = device_index;
        cache.cuda = proj_cpu.to(device, true).contiguous();
    }
    return cache.cuda;
}

static inline void scale_intrinsics_to_image(const int sensor_w,
                                             const int sensor_h,
                                             const int image_w,
                                             const int image_h,
                                             const float fx_in,
                                             const float fy_in,
                                             const float cx_in,
                                             const float cy_in,
                                             float* fx_out,
                                             float* fy_out,
                                             float* cx_out,
                                             float* cy_out,
                                             float* su_out,
                                             float* sv_out)
{
    const float su = (sensor_w > 0) ? static_cast<float>(image_w) / static_cast<float>(sensor_w) : 1.0f;
    const float sv = (sensor_h > 0) ? static_cast<float>(image_h) / static_cast<float>(sensor_h) : 1.0f;

    if (su_out) {
        *su_out = su;
    }
    if (sv_out) {
        *sv_out = sv;
    }
    if (fx_out) {
        *fx_out = fx_in * su;
    }
    if (fy_out) {
        *fy_out = fy_in * sv;
    }
    if (cx_out) {
        *cx_out = (cx_in + 0.5f) * su - 0.5f;
    }
    if (cy_out) {
        *cy_out = (cy_in + 0.5f) * sv - 0.5f;
    }
}

static inline std::vector<int> build_colour_to_depth_index_lut(const int colour_size,
                                                               const int depth_size)
{
    std::vector<int> lut;
    if (colour_size <= 0 || depth_size <= 0) {
        return lut;
    }

    lut.resize(static_cast<size_t>(colour_size));
    const float scale = static_cast<float>(depth_size) / static_cast<float>(colour_size);
    for (int i = 0; i < colour_size; ++i) {
        const int mapped =
            static_cast<int>(std::lround((static_cast<float>(i) + 0.5f) * scale - 0.5f));
        lut[static_cast<size_t>(i)] = std::clamp(mapped, 0, depth_size - 1);
    }
    return lut;
}

struct ColourDepthIndexCache {
    int colour_w = -1;
    int colour_h = -1;
    int depth_w = -1;
    int depth_h = -1;
    bool identity = false;
    std::vector<int> u;
    std::vector<int> v;

    void update(const int new_colour_w,
                const int new_colour_h,
                const int new_depth_w,
                const int new_depth_h)
    {
        if (colour_w == new_colour_w && colour_h == new_colour_h && depth_w == new_depth_w
            && depth_h == new_depth_h) {
            return;
        }
        colour_w = new_colour_w;
        colour_h = new_colour_h;
        depth_w = new_depth_w;
        depth_h = new_depth_h;
        identity = colour_w == depth_w && colour_h == depth_h;
        if (identity) {
            u.clear();
            v.clear();
        }
        else {
            u = build_colour_to_depth_index_lut(colour_w, depth_w);
            v = build_colour_to_depth_index_lut(colour_h, depth_h);
        }
    }
};

static thread_local ColourDepthIndexCache g_colour_depth_index_cache;

struct GsPoseOptimState {
    torch::Tensor theta;
    torch::Tensor rho;
    std::unique_ptr<torch::optim::Adam> optimizer;
};

static inline torch::Tensor skewSymmetric3(const torch::Tensor& v)
{
    torch::Tensor W = torch::zeros({3, 3}, v.options());
    W.index_put_({0, 1}, -v.index({2}));
    W.index_put_({0, 2}, v.index({1}));
    W.index_put_({1, 0}, v.index({2}));
    W.index_put_({1, 2}, -v.index({0}));
    W.index_put_({2, 0}, -v.index({1}));
    W.index_put_({2, 1}, v.index({0}));
    return W;
}

static inline void applyPoseDeltaToCamera(gs::Camera& camera,
                                          const torch::Tensor& rho,
                                          const torch::Tensor& theta,
                                          const torch::Tensor& proj_raw)
{
    if (!rho.defined() || !theta.defined() || rho.numel() != 3 || theta.numel() != 3) {
        return;
    }

    using torch::indexing::Slice;
    torch::NoGradGuard no_grad;
    auto device = camera.T_W2C.device();
    auto dtype = camera.T_W2C.scalar_type();

    auto rho_v = rho.view({3}).to(device, dtype, true).contiguous();
    auto theta_v = theta.view({3}).to(device, dtype, true).contiguous();
    auto I = torch::eye(3, camera.T_W2C.options());
    auto W = skewSymmetric3(theta_v);
    auto W2 = W.mm(W);
    auto angle = torch::norm(theta_v);
    auto angle_safe = torch::clamp(angle, 1.0e-6);

    auto A = torch::sin(angle_safe) / angle_safe;
    auto B = (1.0 - torch::cos(angle_safe)) / (angle_safe * angle_safe);
    auto C = (angle_safe - torch::sin(angle_safe))
             / (angle_safe * angle_safe * angle_safe);

    auto R = I + A * W + B * W2;
    auto V = I + B * W + C * W2;
    auto t = V.mm(rho_v.unsqueeze(1)).squeeze(1);

    auto T_delta = torch::eye(4, camera.T_W2C.options());
    T_delta.index_put_({Slice(0, 3), Slice(0, 3)}, R);
    T_delta.index_put_({Slice(0, 3), 3}, t);

    auto proj_raw_local = proj_raw.to(device, dtype, true).contiguous();
    auto updated_T_W2C = camera.T_W2C.mm(T_delta.transpose(0, 1)).contiguous();
    gs::refreshFixedPoseCameraDerivedTensors(camera, updated_T_W2C, proj_raw_local);
}



static inline int resolveJointTailCount(const int total_iters, const int tail_cfg)
{
    if (total_iters <= 0) {
        return 0;
    }
    if (tail_cfg < 0) {
        return total_iters;
    }
    return std::min(total_iters, std::max(0, tail_cfg));
}

static inline bool extractCameraPoseWorld(const gs::Camera& camera, Eigen::Matrix4f* T_WC)
{
    if (!T_WC) {
        return false;
    }
    auto C2W = camera.T_W2C.inverse().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (!C2W.defined() || C2W.numel() != 16) {
        return false;
    }
    const float* ptr = C2W.data_ptr<float>();
    Eigen::Map<const Eigen::Matrix4f> T_WC_col(ptr);
    *T_WC = T_WC_col;
    return true;
}

} 

namespace se {

template<Colour ColB, int BlockSize, typename SensorT>
GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>, SensorT>::GSUpdater(
    MapType& map,
    const timestamp_t timestamp,
    const Measurements<SensorT>& measurements,
    gs::GaussianModel& gs_model,
    std::vector<gs::Camera>& gs_cam_list,
    std::vector<torch::Tensor>& gt_img_list,
    std::vector<gs::GeometryTarget>& geometry_target_list,
    std::vector<torch::Tensor>& depth_target_list_cpu,
    gs::DataQueue& data_queue,
    std::vector<gs::Camera>* gs_cam_list_cpu,
    std::vector<torch::Tensor>* gt_img_list_cpu,
    gs::PoseUpdateVector* pose_updates,
    gs::FisherTrainingSampleVector* fisher_training_samples,
    gs::DataPacket* preview_packet,
    bool* occupancy_completed) :
        map_(map),
        octree_(map.getOctree()),
        sensor_(measurements.depth.sensor),
        depth_img_(measurements.depth.image),

        T_CW_(measurements.depth.T_WC.inverse()),
        colour_sensor_(measurements.colour ? &measurements.colour->sensor : nullptr),
        colour_img_(measurements.colour ? &measurements.colour->image : nullptr),
        has_colour_(measurements.colour),
        timestamp_(timestamp),
        map_res_(map.getRes()),
        config_(map),
        node_set_(octree_.getBlockDepth()),
        gs_model_(gs_model),
        gs_cam_list_(gs_cam_list),
        gt_img_list_(gt_img_list),
        geometry_target_list_(geometry_target_list),
        depth_target_list_cpu_(depth_target_list_cpu),
        gs_cam_list_cpu_(gs_cam_list_cpu),
        gt_img_list_cpu_(gt_img_list_cpu),
        pose_updates_(pose_updates),
        fisher_training_samples_(fisher_training_samples),
        preview_packet_(preview_packet),
        data_queue_(data_queue),
        occupancy_completed_(occupancy_completed)
{
    if (pose_updates_) {
        pose_updates_->clear();
    }
    if (fisher_training_samples_) {
        fisher_training_samples_->clear();
    }
    if (occupancy_completed_) {
        *occupancy_completed_ = false;
    }


    if (!has_colour_ || !colour_sensor_ || !colour_img_) {
        throw std::runtime_error("GSUpdater requires valid RGB measurement.");
    }

    if constexpr (ColB == Colour::On) {
        if (has_colour_) {
            T_CcC_ = measurements.colour->T_WC.inverse() * measurements.depth.T_WC;
            T_WCc_ = measurements.colour->T_WC;
        }
    }

    scale_intrinsics_to_image(colour_sensor_->model.imageWidth(),
                              colour_sensor_->model.imageHeight(),
                              colour_img_->width(),
                              colour_img_->height(),
                              colour_sensor_->model.focalLengthU(),
                              colour_sensor_->model.focalLengthV(),
                              colour_sensor_->model.imageCenterU(),
                              colour_sensor_->model.imageCenterV(),
                              &colour_fx_,
                              &colour_fy_,
                              &colour_cx_,
                              &colour_cy_,
                              &colour_scale_u_,
                              &colour_scale_v_);

    start_time_ = PerfStats::timeNow();
    if (measurements.rgb_input) {
        rgb_input_ = *measurements.rgb_input;
    }
    else {
        rgb_input_.host_ptr = reinterpret_cast<const uint8_t*>(colour_img_->data());
        rgb_input_.width = colour_img_->width();
        rgb_input_.height = colour_img_->height();
        rgb_input_.row_stride = static_cast<size_t>(colour_img_->width()) * 3u;
        rgb_input_.memory_kind = gs::RgbInputMemoryKind::OrdinaryHost;
        rgb_input_.channel_order = gs::RgbChannelOrder::RGB;
    }
    if (!rgb_input_.valid()
        || rgb_input_.width != colour_img_->width()
        || rgb_input_.height != colour_img_->height()) {
        throw std::runtime_error("GSUpdater RGB input contract does not match colour image");
    }

    const Eigen::Matrix4f T_WCc_matrix = T_WCc_.matrix();
    const Eigen::Matrix3f R_WCc = T_WCc_.linear();
    const float rotation_error =
        (R_WCc.transpose() * R_WCc - Eigen::Matrix3f::Identity()).norm();
    const float determinant = R_WCc.determinant();
    if (!T_WCc_matrix.allFinite() || !std::isfinite(rotation_error)
        || !std::isfinite(determinant)
        || rotation_error > 1.0e-3f
        || std::abs(determinant - 1.0f) > 1.0e-3f
        || std::abs(T_WCc_matrix(3, 0)) > 1.0e-6f
        || std::abs(T_WCc_matrix(3, 1)) > 1.0e-6f
        || std::abs(T_WCc_matrix(3, 2)) > 1.0e-6f
        || std::abs(T_WCc_matrix(3, 3) - 1.0f) > 1.0e-6f) {
        throw std::runtime_error("GSUpdater received an invalid RGB camera pose");
    }

    Eigen::Matrix4f T_CcW = T_WCc_.inverse().matrix();
    torch::Tensor W2C_cpu =
        torch::from_blob(T_CcW.data(), {4, 4}, torch::kFloat)
            .clone()
            .to(torch::kCPU)
            .contiguous();

    const torch::Tensor& proj_cpu = getCachedProjectionMatrixCpu(colour_img_->width(),
                                                                 colour_img_->height(),
                                                                 colour_fx_,
                                                                 colour_fy_,
                                                                 colour_cx_,
                                                                 colour_cy_);

    const bool retain_cpu_camera = gs_cam_list_cpu_ != nullptr;
    gs::Camera cam_cpu;
    if (retain_cpu_camera) {
        cam_cpu.width = colour_img_->width();
        cam_cpu.height = colour_img_->height();
        cam_cpu.fov_x = colour_sensor_->horizontal_fov;
        cam_cpu.fov_y = colour_sensor_->vertical_fov;
        cam_cpu.fx = colour_fx_;
        cam_cpu.fy = colour_fy_;
        cam_cpu.cx = colour_cx_;
        cam_cpu.cy = colour_cy_;
        gs::refreshFixedPoseCameraDerivedTensors(cam_cpu, W2C_cpu, proj_cpu);
    }

    auto rgb_ingest_result = gs::ingestRgb(rgb_input_);
    cur_gt_img_ = std::move(rgb_ingest_result.image);
    cur_gt_img_host_cpu_ = std::move(rgb_ingest_result.host_cpu_image);

    torch::Tensor W2C_matrix = W2C_cpu.to(cur_gt_img_.device(), true);

    torch::Tensor proj_matrix = getCachedProjectionMatrixCuda(colour_img_->width(),
                                                              colour_img_->height(),
                                                              colour_fx_,
                                                              colour_fy_,
                                                              colour_cx_,
                                                              colour_cy_,
                                                              cur_gt_img_.device());

    cur_gs_cam_.width = colour_img_->width();
    cur_gs_cam_.height = colour_img_->height();
    cur_gs_cam_.fov_x = colour_sensor_->horizontal_fov;
    cur_gs_cam_.fov_y = colour_sensor_->vertical_fov;
    cur_gs_cam_.fx = colour_fx_;
    cur_gs_cam_.fy = colour_fy_;
    cur_gs_cam_.cx = colour_cx_;
    cur_gs_cam_.cy = colour_cy_;
    gs::refreshFixedPoseCameraDerivedTensors(cur_gs_cam_, W2C_matrix, proj_matrix);

    if (gs_model_.optimParams.geometrySupervisionActive()) {
        if (!depth_img_.data() || depth_img_.width() != colour_img_->width()
            || depth_img_.height() != colour_img_->height()) {
            throw std::runtime_error(
                "Geometry supervision requires input depth aligned to RGB");
        }
        cur_depth_target_cpu_ = gs::makeDepthCpuSnapshot(depth_img_.data(),
                                                         depth_img_.width(),
                                                         depth_img_.height(),
                                                         sensor_.near_plane,
                                                         sensor_.far_plane);
        cur_geometry_target_ = gs::makeGeometryTarget(cur_depth_target_cpu_,
                                                       cur_gs_cam_,
                                                       0.0f,
                                                       std::numeric_limits<float>::infinity(),
                                                       cur_gt_img_.device());
    }

    gt_img_list_.push_back(cur_gt_img_);
    gs_cam_list_.push_back(cur_gs_cam_);
    if (gs_model_.optimParams.geometrySupervisionActive()) {
        geometry_target_list_.push_back(cur_geometry_target_);
    }
    if (retain_cpu_camera) {
        gs_cam_list_cpu_->push_back(std::move(cam_cpu));
    }

    if (preview_packet_) {

        cv::Mat cv_src_depth(depth_img_.height(),
                             depth_img_.width(),
                             CV_32FC1,
                             (void*) depth_img_.data());
        cv::Mat depth_vis_u8(cv_src_depth.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat invalid_mask(cv_src_depth.size(), CV_8UC1, cv::Scalar(255));

        const float min_depth = std::max(sensor_.near_plane, 0.0f);
        const float max_depth = (sensor_.far_plane > min_depth + 1.0e-3f)
                                    ? sensor_.far_plane
                                    : (min_depth + 1.0f);
        const float inv_span = 255.0f / (max_depth - min_depth);

        for (int v = 0; v < cv_src_depth.rows; ++v) {
            const float* src_row = cv_src_depth.ptr<float>(v);
            uint8_t* dst_row = depth_vis_u8.ptr<uint8_t>(v);
            uint8_t* mask_row = invalid_mask.ptr<uint8_t>(v);
            for (int u = 0; u < cv_src_depth.cols; ++u) {
                const float d = src_row[u];
                if (std::isfinite(d) && d > min_depth && d < max_depth) {
                    const float scaled =
                        std::clamp((d - min_depth) * inv_span, 0.0f, 255.0f);
                    dst_row[u] = static_cast<uint8_t>(scaled);
                    mask_row[u] = 0;
                }
            }
        }

        cv::Mat cv_depth_color;
        cv::applyColorMap(depth_vis_u8, cv_depth_color, cv::COLORMAP_VIRIDIS);
        cv_depth_color.setTo(cv::Scalar(0, 0, 0), invalid_mask);
        cv::cvtColor(cv_depth_color, cv_depth_color, cv::COLOR_BGR2RGB);
        data_packet_.depth = std::move(cv_depth_color);
    }

}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>, SensorT>::operator()(
    VolumeCarverAllocation& allocation_list,
    std::vector<const OctantBase*>* updated_octants)
{
    GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/frame");
    cv::Mat cv_src_img(colour_img_->height(),
                       colour_img_->width(),
                       CV_8UC3,
                       const_cast<colour_t*>(colour_img_->data()));

#ifdef _OPENMP
    const int default_qtree_threads = std::min(2, std::max(1, omp_get_max_threads()));
#else
    const int default_qtree_threads = 1;
#endif
    const int qtree_threads =
        env_int_clamped("GOFUSION_QTREE_THREADS", default_qtree_threads, 1, 16);
    const int qtree_split_depth = env_int_clamped("GOFUSION_QTREE_SPLIT_DEPTH", 3, 0, 4);
    const bool qtree_overlap = env_bool("GOFUSION_QTREE_OVERLAP", true);
    static thread_local gs::QTreeWorker qtree_worker;
    if (qtree_overlap) {
        qtree_worker.submit(cv_src_img,
                            gs_model_.optimParams.qtree_thresh,
                            gs_model_.optimParams.qtree_min_pixel_size,
                            gs_model_.optimParams.qtree_ring_thresh,
                            gs_model_.optimParams.qtree_ring_t,
                            qtree_threads,
                            qtree_split_depth,
                            rgb_input_.owner);
    }

    static std::atomic<bool> qtree_config_logged{false};
    if (!qtree_config_logged.exchange(true)) {
        LOG(INFO) << "GS quadtree CPU pipeline: overlap=" << qtree_overlap
                  << " threads=" << qtree_threads
                  << " split_depth=" << qtree_split_depth;
    }
    gs::profiling::NvtxRange occupancy_range("GOFusion/GSUpdater/occupancy");

    TICK("fusion-total")

    TICK("fusion-nodes")

    for (auto& v : node_set_) {
        v.clear();
    }
    freed_block_list_.clear();

#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
#else
    const int num_threads = 1;
#endif

    struct ThreadScratch {
        std::vector<std::vector<std::vector<OctantBase*>>> tls_node_set;
        std::vector<std::vector<OctantBase*>> tls_freed_blocks;
    };
    static thread_local ThreadScratch scratch;
    if (scratch.tls_node_set.size() < static_cast<size_t>(num_threads)) {
        scratch.tls_node_set.resize(num_threads);
    }
    if (scratch.tls_freed_blocks.size() < static_cast<size_t>(num_threads)) {
        scratch.tls_freed_blocks.resize(num_threads);
    }
    auto& tls_node_set = scratch.tls_node_set;
    auto& tls_freed_blocks = scratch.tls_freed_blocks;
    for (int t = 0; t < num_threads; ++t) {
        auto& per_thread_node_set = tls_node_set[t];
        if (per_thread_node_set.size() != node_set_.size()) {
            per_thread_node_set.resize(node_set_.size());
        }
        for (auto& level : per_thread_node_set) {
            level.clear();
        }
        tls_freed_blocks[t].clear();
    }

#pragma omp parallel for
    for (unsigned int i = 0; i < allocation_list.node_list.size(); ++i) {
        auto* node_ptr = static_cast<NodeType*>(allocation_list.node_list[i]);
        const int depth = octree_.getMaxScale() - std::log2(node_ptr->size);

#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        freeNodeRecurse(allocation_list.node_list[i], depth, tls_node_set[tid], tls_freed_blocks[tid]);
    }

    for (int t = 0; t < num_threads; ++t) {
        for (size_t d = 0; d < node_set_.size(); ++d) {
            auto& src = tls_node_set[t][d];
            if (!src.empty()) {
                auto& dst = node_set_[d];
                dst.insert(dst.end(), src.begin(), src.end());
            }
        }
        auto& fb = tls_freed_blocks[t];
        if (!fb.empty()) {
            freed_block_list_.insert(freed_block_list_.end(), fb.begin(), fb.end());
        }
    }

    TOCK("fusion-nodes")

    TICK("fusion-blocks")
#pragma omp parallel for
    for (unsigned int i = 0; i < allocation_list.block_list.size(); ++i) {
        updateBlock(allocation_list.block_list[i],
                    allocation_list.variance_state_list[i],
                    allocation_list.projects_inside_list[i]);
    }
    TOCK("fusion-blocks")

    TOCK("fusion-total")

    TICK("propagation-total")

    TICK("propagation-blocks")
#pragma omp parallel for
    for (unsigned int i = 0; i < allocation_list.block_list.size(); ++i) {
        updater::propagate_block_to_coarsest_scale<BlockType>(allocation_list.block_list[i]);
    }
#pragma omp parallel for
    for (unsigned int i = 0; i < freed_block_list_.size(); ++i) {
        updater::propagate_block_to_coarsest_scale<BlockType>(freed_block_list_[i]);
    }
    TOCK("propagation-blocks")

    track_updated_octants_ = updated_octants;
    if (track_updated_octants_) {
        updated_octants_.clear();
        updated_octants_.insert(allocation_list.block_list.begin(),
                                allocation_list.block_list.end());
        updated_octants_.insert(freed_block_list_.begin(), freed_block_list_.end());
        std::copy_if(allocation_list.node_list.begin(),
                     allocation_list.node_list.end(),
                     std::inserter(updated_octants_, updated_octants_.end()),
                     [](const auto octant) { return octant && octant->isLeaf(); });
    }

    TICK("propagation-to-root")
    propagateToRoot(allocation_list.block_list);
    TOCK("propagation-to-root")

    TOCK("propagation-total")

    if (track_updated_octants_) {
        updated_octants->clear();
        updated_octants->reserve(updated_octants_.size());
        updated_octants->insert(
            updated_octants->end(), updated_octants_.begin(), updated_octants_.end());
    }
    if (occupancy_completed_) {
        *occupancy_completed_ = true;
    }
    occupancy_range.stop();

    gs::profiling::NvtxRange frame_prepare_range("GOFusion/GSUpdater/frame_prepare");
    if (preview_packet_) {
        data_packet_.rgb = cv_src_img.clone();
    }

    if (!qtree_overlap) {
        qtree_worker.submit(cv_src_img,
                            gs_model_.optimParams.qtree_thresh,
                            gs_model_.optimParams.qtree_min_pixel_size,
                            gs_model_.optimParams.qtree_ring_thresh,
                            gs_model_.optimParams.qtree_ring_t,
                            qtree_threads,
                            qtree_split_depth,
                            rgb_input_.owner);
    }
    gs::profiling::NvtxRange qtree_join_range("GOFusion/GSUpdater/qtree_join");
    const std::vector<gs::Leaf>& nodes = qtree_worker.wait();
    qtree_join_range.stop();

    const auto& densify_cfg = gs_model_.densify_params;
    const bool do_densify = densify_cfg.enabled && densify_cfg.interval > 0
                            && (timestamp_ % densify_cfg.interval == 0);

    if (do_densify) {
        const int W = colour_img_->width();
        const int H = colour_img_->height();

        g_qtree_leaf_W = W;
        g_qtree_leaf_H = H;
        g_qtree_leaf_count = static_cast<int>(nodes.size());

        g_qtree_leaves.assign(nodes.begin(), nodes.end());
        gs::build_leaf_id_lut(nodes, W, H, g_qtree_leaf_id, qtree_threads);
    }

    auto& candidate_scratch = g_gaussian_candidate_scratch;
    auto& positions = candidate_scratch.positions;
    auto& colors = candidate_scratch.colors;
    auto& scales_xyz = candidate_scratch.scales_xyz;
    auto& rots = candidate_scratch.rotations_wxyz;
    const size_t num_nodes = nodes.size();
    candidate_scratch.candidates.resize(num_nodes);
    candidate_scratch.accepted.assign(num_nodes, uint8_t(0));

    const float depth_fx = sensor_.model.focalLengthU();
    const float depth_fy = sensor_.model.focalLengthV();
    const float depth_cx = sensor_.model.imageCenterU();
    const float depth_cy = sensor_.model.imageCenterV();
    const float inv_depth_fx = 1.0f / depth_fx;
    const float inv_depth_fy = 1.0f / depth_fy;

    const int W = colour_img_->width();
    const int H = colour_img_->height();
    const int depthW = depth_img_.width();
    const int depthH = depth_img_.height();

    const float colour_to_depth_scale_x = static_cast<float>(depthW) / static_cast<float>(W);
    const float colour_to_depth_scale_y = static_cast<float>(depthH) / static_cast<float>(H);
    const float* depth_ptr = depth_img_.data();
    g_colour_depth_index_cache.update(W, H, depthW, depthH);
    const auto& colour_depth_lut = g_colour_depth_index_cache;

    const Eigen::Matrix3f R_WCc = T_WCc_.linear();
    const Eigen::Vector3f t_WCc = T_WCc_.translation();
    const Eigen::Vector3f exW = R_WCc.col(0);
    const Eigen::Vector3f eyW = R_WCc.col(1);
    const int gaussian_init_threads = adaptive_parallel_threads(num_nodes);

    {
        GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/gaussian_center_validate");
#pragma omp parallel for schedule(static) num_threads(gaussian_init_threads) if(gaussian_init_threads > 1)
        for (int i = 0; i < static_cast<int>(num_nodes); i++) {
            const gs::Leaf& node = nodes[static_cast<size_t>(i)];

            const Eigen::Vector2f p2d(node.getOriginX() + 0.5f * node.getWidth(),
                                      node.getOriginY() + 0.5f * node.getHeight());
            const Eigen::Vector2i pixel = se::round_pixel(p2d);

            const int u = pixel.x();
            const int v = pixel.y();

            if (u < 0 || u >= W || v < 0 || v >= H) {
                continue;
            }

            const int u_depth = colour_depth_lut.identity
                                    ? u
                                    : colour_depth_lut.u[static_cast<size_t>(u)];
            const int v_depth = colour_depth_lut.identity
                                    ? v
                                    : colour_depth_lut.v[static_cast<size_t>(v)];

            const float depth_m = depth_ptr[u_depth + depthW * v_depth];
            const bool depth_in_range =
                std::isfinite(depth_m)
                && (depth_m > sensor_.near_plane)
                && (depth_m < sensor_.far_plane);
            if (!depth_in_range) {
                continue;
            }

            auto& candidate = candidate_scratch.candidates[static_cast<size_t>(i)];
            if (!gs::make_gaussian_center_from_valid_depth_pixel_precomp(&candidate.pos_w,
                                                                         u_depth,
                                                                         v_depth,
                                                                         depth_m,
                                                                         inv_depth_fx,
                                                                         inv_depth_fy,
                                                                         depth_cx,
                                                                         depth_cy,
                                                                         R_WCc,
                                                                         t_WCc)) {
                continue;
            }

            const Eigen::Vector3f center(
                candidate.pos_w.x, candidate.pos_w.y, candidate.pos_w.z);

            auto center_data = map_.template getData<se::Safe::On>(center);
            if (center_data.field.weight != 1) {
                continue;
            }

            candidate.depth_m = depth_m;
            candidate.u_colour = u;
            candidate.v_colour = v;
            candidate.u_depth = u_depth;
            candidate.v_depth = v_depth;
            candidate.leaf_w_depth_px =
                static_cast<float>(node.getWidth()) * colour_to_depth_scale_x;
            candidate.leaf_h_depth_px =
                static_cast<float>(node.getHeight()) * colour_to_depth_scale_y;
            candidate_scratch.accepted[static_cast<size_t>(i)] = uint8_t(1);
        }
    }

    size_t valid_count = 0;
    {
        GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/gaussian_stable_compaction");
        valid_count = stable_compact_indices(candidate_scratch.accepted,
                                              candidate_scratch.accepted_indices,
                                              candidate_scratch.thread_counts,
                                              candidate_scratch.thread_offsets,
                                              gaussian_init_threads);
    }
    positions.resize(valid_count);
    colors.resize(valid_count);
    scales_xyz.resize(valid_count);
    rots.resize(valid_count);

    const int gaussian_finish_threads = adaptive_parallel_threads(valid_count);
    {
        GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/gaussian_finish_init");
#pragma omp parallel for schedule(static) num_threads(gaussian_finish_threads) if(gaussian_finish_threads > 1)
        for (int output_idx = 0; output_idx < static_cast<int>(valid_count); ++output_idx) {
            const size_t leaf_idx = static_cast<size_t>(
                candidate_scratch.accepted_indices[static_cast<size_t>(output_idx)]);
            const auto& candidate = candidate_scratch.candidates[leaf_idx];
            gs::OrientedGaussianInit init{};
            const bool initialized = gs::finish_oriented_gaussian_from_depth_pixel_precomp(
                &init,
                candidate.pos_w,
                candidate.u_depth,
                candidate.v_depth,
                candidate.leaf_w_depth_px,
                candidate.leaf_h_depth_px,
                candidate.depth_m,
                depth_ptr,
                depthW,
                depthH,
                sensor_.near_plane,
                inv_depth_fx,
                inv_depth_fy,
                depth_cx,
                depth_cy,
                R_WCc,
                exW,
                eyW,
                densify_cfg.init_scale_mul,
                densify_cfg.init_z_scale);
            assert(initialized);
            (void) initialized;

            const size_t out = static_cast<size_t>(output_idx);
            positions[out] = init.pos_w;
            scales_xyz[out] = init.scale_xyz;
            rots[out] = init.rot_wxyz;

            const auto rgb = (*colour_img_)[candidate.u_colour + W * candidate.v_colour];
            colors[out] = gs::Color{rgb.r, rgb.g, rgb.b};
        }
    }

    static const bool qtree_stats_enabled = env_bool("GOFUSION_QTREE_STATS", false);
    static std::atomic<uint64_t> qtree_stats_frame{0};
    const uint64_t stats_frame = qtree_stats_enabled
                                     ? qtree_stats_frame.fetch_add(1, std::memory_order_relaxed) + 1
                                     : 0;
    if (qtree_stats_enabled && (stats_frame % 120 == 0)) {
        const double accepted_percent = num_nodes == 0
                                            ? 0.0
                                            : 100.0 * static_cast<double>(valid_count)
                                                  / static_cast<double>(num_nodes);
        LOG(INFO) << "GS quadtree candidates: leaves=" << num_nodes
                  << " accepted=" << valid_count
                  << " accepted_percent=" << accepted_percent
                  << " validate_threads=" << gaussian_init_threads
                  << " finish_threads=" << gaussian_finish_threads;
    }

    if (valid_count > static_cast<size_t>(gs_model_.optimParams.kf_thresh)) {
        isKeyframe_ = true;
    }
    else {
        if (!gs_model_.optimParams.keep_all_frames) {
            gs_cam_list_.pop_back();
            gt_img_list_.pop_back();
            if (gs_model_.optimParams.geometrySupervisionActive()) {
                geometry_target_list_.pop_back();
            }
            if (gs_cam_list_cpu_) {
                gs_cam_list_cpu_->pop_back();
            }
        }
    }

    if ((isKeyframe_ || gs_model_.optimParams.keep_all_frames)
        && gt_img_list_cpu_) {
        if (cur_gt_img_host_cpu_.defined()) {
            gt_img_list_cpu_->push_back(
                cur_gt_img_host_cpu_.to(torch::TensorOptions()
                                                .dtype(torch::kFloat16)
                                                .device(torch::kCPU)
                                                .pinned_memory(true)));
        }
        else {
            gt_img_list_cpu_->push_back(gs::makeRgbCpuSnapshot(cur_gt_img_));
        }
    }
    if ((isKeyframe_ || gs_model_.optimParams.keep_all_frames)
        && gs_model_.optimParams.geometrySupervisionActive()) {
        depth_target_list_cpu_.push_back(cur_depth_target_cpu_);
    }
    cur_gt_img_host_cpu_ = torch::Tensor();

    if (rgb_input_.release_owner) {
        rgb_input_.release_owner();
        rgb_input_.release_owner = {};
    }

    frame_prepare_range.stop();
    updateGSModel(positions, colors, scales_xyz, rots);

    if (rgb_input_.release_owner) {
        rgb_input_.release_owner();
        rgb_input_.release_owner = {};
    }

}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>,
               SensorT>::propagateToRoot(std::vector<OctantBase*>& block_list)
{
    const int block_depth = octree_.getBlockDepth();

    for (const auto& octant_ptr : block_list) {
        auto* block_ptr = static_cast<BlockType*>(octant_ptr);
        if (block_ptr->parent()) {
            node_set_[block_depth - 1].push_back(block_ptr->parent());
        }
    }

    for (int d = block_depth - 1; d > 0; --d) {
        auto& level = node_set_[d];
        if (!level.empty()) {
            std::sort(level.begin(), level.end());
            level.erase(std::unique(level.begin(), level.end()), level.end());
        }

        for (OctantBase* const octant_ptr : level) {
            if (octant_ptr->timestamp == timestamp_) {
                continue;
            }
            if (octant_ptr->parent()) {
                auto node_data =
                    updater::propagate_to_parent_node<NodeType, BlockType>(octant_ptr, timestamp_);
                node_set_[d - 1].push_back(octant_ptr->parent());
                if (track_updated_octants_) {
                    updated_octants_.insert(octant_ptr);
                }

                if (node_data.field.observed
                    && get_field(node_data) <= 0.95 * MapType::DataType::FieldType::min_occupancy) {
                    auto* const node_ptr = static_cast<NodeType*>(octant_ptr);
                    if (track_updated_octants_) {
                        for (int i = 0; i < 8; i++) {
                            OctantBase* const child_ptr = node_ptr->getChild(i);
                            if (child_ptr) {
                                updated_octants_.erase(child_ptr);
                            }
                        }
                    }
                    octree_.deleteChildren(node_ptr);
                }
            }
        }
    }

    updater::propagate_to_parent_node<NodeType, BlockType>(octree_.getRoot(), timestamp_);
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>, SensorT>::freeBlock(
    OctantBase* octant_ptr)
{
    assert(octant_ptr);
    assert(octant_ptr->is_block);

    BlockType* block_ptr = static_cast<BlockType*>(octant_ptr);

    const unsigned int block_size = BlockType::size;
    const Eigen::Vector3i block_coord = block_ptr->coord;
    Eigen::Vector3f block_centre_point_W;

    map_.voxelToPoint(block_coord, block_size, block_centre_point_W);
    const Eigen::Vector3f block_centre_point_C = T_CW_ * block_centre_point_W;

    const int last_scale = (block_ptr->min_scale == -1) ? 0 : block_ptr->current_scale;

    const int computed_integration_scale =
        sensor_.blockIntegrationScale(block_centre_point_C,
                                        map_res_,
                                        last_scale,
                                        block_ptr->min_scale,
                                        block_ptr->max_scale);

    const auto& data_config = map_.getDataConfig();
    const auto& field_cfg = data_config.field;
    const int min_integration_scale = ((block_ptr->min_scale == -1
                                        || block_ptr->maxData().field.occupancy
                                            < 0.95 * field_cfg.log_odd_min))
        ? field_cfg.fs_integr_scale
        : std::max(0, last_scale - 1);
    const int max_integration_scale = (block_ptr->min_scale == -1)
        ? BlockType::max_scale
        : std::min(BlockType::max_scale, last_scale + 1);

    const int recommended_scale = std::min(
        std::max(min_integration_scale, computed_integration_scale), max_integration_scale);

    int integration_scale = last_scale;

    if (block_ptr->min_scale == -1) {

        integration_scale = recommended_scale;
        block_ptr->allocateDownTo(integration_scale);
        block_ptr->current_scale = (integration_scale);
        block_ptr->initCurrCount();
        block_ptr->init_data = (DataType());
    }
    else if (recommended_scale != last_scale) 
    {
        if (recommended_scale
            != block_ptr->bufferScale()) 
        {
            block_ptr->initBuffer(recommended_scale);

            if (recommended_scale < last_scale) {
                const int parent_scale = last_scale;
                const unsigned int size_at_parent_scale_li = block_size >> parent_scale;
                const unsigned int size_at_parent_scale_sq = math::sq(size_at_parent_scale_li);
                const auto* const parent_data_at_parent_scale =
                    block_ptr->blockDataAtScale(parent_scale);

                const unsigned int size_at_buffer_scale_li = size_at_parent_scale_li << 1;
                const unsigned int size_at_buffer_scale_sq = math::sq(size_at_buffer_scale_li);

                for (unsigned int z = 0; z < size_at_parent_scale_li; z++) {
                    for (unsigned int y = 0; y < size_at_parent_scale_li; y++) {
                        for (unsigned int x = 0; x < size_at_parent_scale_li; x++) {
                            const int parent_idx =
                                x + y * size_at_parent_scale_li + z * size_at_parent_scale_sq;
                            const auto& parent_data = parent_data_at_parent_scale[parent_idx];

                            for (unsigned int k = 0; k < 2; k++) {
                                for (unsigned int j = 0; j < 2; j++) {
                                    for (unsigned int i = 0; i < 2; i++) {
                                        const int buffer_idx = (2 * x + i)
                                            + (2 * y + j) * size_at_buffer_scale_li
                                            + (2 * z + k) * size_at_buffer_scale_sq;
                                        auto& buffer_data = block_ptr->bufferData(buffer_idx);

                                        buffer_data.field.occupancy = parent_data.field.occupancy;
                                        buffer_data.field.weight = parent_data.field.weight;
                                        buffer_data.field.observed =
                                            false; 

                                    } 
                                }     
                            }         

                        } 
                    }     
                }         
            }
        }

        const unsigned int size_at_recommended_scale_li = BlockType::size >> recommended_scale;
        const unsigned int size_at_recommended_scale_sq = math::sq(size_at_recommended_scale_li);

        for (unsigned int z = 0; z < size_at_recommended_scale_li; z++) {
            for (unsigned int y = 0; y < size_at_recommended_scale_li; y++) {
                for (unsigned int x = 0; x < size_at_recommended_scale_li; x++) {
                    const int buffer_idx =
                        x + y * size_at_recommended_scale_li + z * size_at_recommended_scale_sq;
                    auto& buffer_data = block_ptr->bufferData(buffer_idx);
                    block_ptr->incrBufferObservedCount(
                        updater::free_voxel(buffer_data, data_config));

                } 
            }     
        }         

        block_ptr->incrBufferIntegrCount();

        if (block_ptr->switchData()) {
            return;
        }
    }
    else {
        block_ptr->resetBuffer();
    }

    const unsigned int size_at_integration_scale_li = BlockType::size >> integration_scale;
    const unsigned int size_at_integration_scale_sq = math::sq(size_at_integration_scale_li);
    auto* const data_at_integration_scale = block_ptr->blockDataAtScale(integration_scale);

    for (unsigned int z = 0; z < size_at_integration_scale_li; z++) {
        for (unsigned int y = 0; y < size_at_integration_scale_li; y++) {
            for (unsigned int x = 0; x < size_at_integration_scale_li; x++) {
                const int voxel_idx =
                    x + y * size_at_integration_scale_li + z * size_at_integration_scale_sq;
                auto& voxel_data = data_at_integration_scale[voxel_idx];
                block_ptr->incrCurrObservedCount(
                    updater::free_voxel(voxel_data, data_config));

            } 
        }     
    }         

    block_ptr->incrCurrIntegrCount();
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>,
               SensorT>::updateBlock(
    OctantBase* octant_ptr,
    const VarianceState variance_state,
    bool project_inside)
{
    assert(octant_ptr);
    assert(octant_ptr->is_block);

    if (variance_state == VarianceState::Undefined) {
        return;
    }
    if (variance_state == VarianceState::Constant) {
        freeBlock(octant_ptr);
        return;
    }

    const auto& data_config = map_.getDataConfig();
    const auto& field_cfg = data_config.field;
    const auto three_sigma_from_depth = [&](const float depth_value) -> float {
        float sigma_value = field_cfg.k_sigma * depth_value;
        if (field_cfg.uncertainty_model == UncertaintyModel::Quadratic) {
            sigma_value *= depth_value;
        }
        sigma_value = std::clamp(sigma_value, config_.sigma_min, config_.sigma_max);
        return 3.0f * sigma_value;
    };

    BlockType* block_ptr = static_cast<BlockType*>(octant_ptr);
    const int block_size = BlockType::size;
    const Eigen::Vector3i block_coord = block_ptr->coord;

    Eigen::Vector3f block_centre_point_W;
    map_.voxelToPoint(block_coord, block_size, block_centre_point_W);
    const Eigen::Vector3f block_centre_point_C = T_CW_ * block_centre_point_W;

    const float block_point_C_m = sensor_.measurementFromPoint(block_centre_point_C);
    const float tau = compute_tau(block_point_C_m, config_.tau_min, config_.tau_max, data_config);

    const int last_scale = (block_ptr->min_scale == -1) ? 0 : block_ptr->current_scale;

    const int computed_integration_scale =
        sensor_.blockIntegrationScale(block_centre_point_C,
                                      map_res_,
                                      last_scale,
                                      block_ptr->min_scale,
                                      block_ptr->max_scale);

    const int min_integration_scale = std::max(0, last_scale - 1);
    const int max_integration_scale = (block_ptr->min_scale == -1)
        ? BlockType::max_scale
        : std::min(BlockType::max_scale, last_scale + 1);

    const int recommended_scale = std::min(
        std::max(min_integration_scale, computed_integration_scale), max_integration_scale);

    int integration_scale = last_scale;

    if (block_ptr->min_scale == -1) {
        integration_scale = recommended_scale;
        block_ptr->allocateDownTo(integration_scale);
        block_ptr->current_scale = integration_scale;
        block_ptr->initCurrCount();
        block_ptr->init_data = (DataType());
    }
    else if (recommended_scale != last_scale) {
        if (recommended_scale != block_ptr->bufferScale()) {
            block_ptr->initBuffer(recommended_scale);

            if (recommended_scale < last_scale) {
                const int parent_scale = last_scale;
                const unsigned int size_at_parent_scale_li = BlockType::size >> parent_scale;
                const unsigned int size_at_parent_scale_sq = math::sq(size_at_parent_scale_li);
                const auto* const parent_data_at_parent_scale =
                    block_ptr->blockDataAtScale(parent_scale);

                const unsigned int size_at_buffer_scale_li = size_at_parent_scale_li << 1;
                const unsigned int size_at_buffer_scale_sq = math::sq(size_at_buffer_scale_li);

                for (unsigned int z = 0; z < size_at_parent_scale_li; z++) {
                    for (unsigned int y = 0; y < size_at_parent_scale_li; y++) {
                        for (unsigned int x = 0; x < size_at_parent_scale_li; x++) {
                            const int parent_idx =
                                x + y * size_at_parent_scale_li + z * size_at_parent_scale_sq;
                            const auto& parent_data = parent_data_at_parent_scale[parent_idx];

                            for (unsigned int k = 0; k < 2; k++) {
                                for (unsigned int j = 0; j < 2; j++) {
                                    for (unsigned int i = 0; i < 2; i++) {
                                        const int buffer_idx = (2 * x + i)
                                            + (2 * y + j) * size_at_buffer_scale_li
                                            + (2 * z + k) * size_at_buffer_scale_sq;
                                        auto& buffer_data = block_ptr->bufferData(buffer_idx);

                                        buffer_data.field.occupancy = parent_data.field.occupancy;
                                        buffer_data.field.weight = parent_data.field.weight;
                                        buffer_data.field.observed = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        const unsigned int recommended_stride = 1 << recommended_scale;
        const unsigned int size_at_recommended_scale_li = BlockType::size >> recommended_scale;
        const unsigned int size_at_recommended_scale_sq = math::sq(size_at_recommended_scale_li);

        const Eigen::Vector3i voxel_coord_base = block_ptr->coord;
        Eigen::Vector3f sample_point_base_W;
        map_.voxelToPoint(voxel_coord_base, recommended_stride, sample_point_base_W);
        const Eigen::Vector3f sample_point_base_C = T_CW_ * sample_point_base_W;
        const int colour_width = has_colour_ ? colour_img_->width() : 0;
        const int colour_height = has_colour_ ? colour_img_->height() : 0;
        const float max_weight = data_config.field.max_weight;

        const Eigen::Matrix3f sample_point_delta_matrix_C =
            (T_CW_.linear()
             * (map_res_
                * (Eigen::Matrix3f() << recommended_stride,
                   0,
                   0,
                   0,
                   recommended_stride,
                   0,
                   0,
                   0,
                   recommended_stride)
                      .finished()));

        for (unsigned int z = 0; z < size_at_recommended_scale_li; z++) {
            for (unsigned int y = 0; y < size_at_recommended_scale_li; y++) {
                for (unsigned int x = 0; x < size_at_recommended_scale_li; x++) {
                    const Eigen::Vector3f sample_point_C = sample_point_base_C
                        + sample_point_delta_matrix_C * Eigen::Vector3f(x, y, z);

                    Eigen::Vector2f depth_pixel_f;
                    if (sensor_.model.project(sample_point_C, &depth_pixel_f)
                        != srl::projection::ProjectionStatus::Successful) {
                        continue;
                    }

                    const Eigen::Vector2i depth_pixel = se::round_pixel(depth_pixel_f);
                    const float depth_value = depth_img_(depth_pixel.x(), depth_pixel.y());
                    if (depth_value < sensor_.near_plane) {
                        continue;
                    }

                    const float three_sigma = three_sigma_from_depth(depth_value);

                    const int buffer_idx =
                        x + y * size_at_recommended_scale_li + z * size_at_recommended_scale_sq;
                    auto& buffer_data = block_ptr->bufferData(buffer_idx);

                    const float sample_point_C_m = sensor_.measurementFromPoint(sample_point_C);
                    const float range = sample_point_C.norm();
                    const float range_diff =
                        (sample_point_C_m - depth_value) * (range / sample_point_C_m);
                    block_ptr->incrBufferObservedCount(updater::update_voxel(
                        buffer_data, range_diff, tau, three_sigma, data_config));
                    const bool field_updated = range_diff < tau;

                    Eigen::Vector3f hit_C;
                    if constexpr (ColB == Colour::On) {
                        if (has_colour_ && field_updated) {
                            sensor_.model.backProject(depth_pixel_f, &hit_C);
                            hit_C.array() *= depth_value;
                        }
                    }

                    if constexpr (ColB == Colour::On) {
                        if (has_colour_ && field_updated) {
                            const Eigen::Vector3f hit_Cc = T_CcC_ * hit_C;
                            Eigen::Vector2f colour_pixel_f;
                            if (colour_sensor_->model.project(hit_Cc, &colour_pixel_f)
                                == srl::projection::ProjectionStatus::Successful) {
                                const int u = static_cast<int>(std::lround(
                                    (colour_pixel_f.x() + 0.5f) * colour_scale_u_ - 0.5f));
                                const int v = static_cast<int>(std::lround(
                                    (colour_pixel_f.y() + 0.5f) * colour_scale_v_ - 0.5f));
                                if (u >= 0 && u < colour_width && v >= 0 && v < colour_height) {
                                    buffer_data.colour.update(
                                        (*colour_img_)(u, v), max_weight);
                                }
                            }
                        }
                    }
                }
            }
        }

        block_ptr->incrBufferIntegrCount(project_inside);

        if (block_ptr->switchData()) {
            return;
        }
    }
    else {
        block_ptr->resetBuffer();
    }

    const unsigned int integration_stride = 1 << integration_scale;
    const unsigned int size_at_integration_scale_li = BlockType::size >> integration_scale;
    const unsigned int size_at_integration_scale_sq = math::sq(size_at_integration_scale_li);
    auto* const data_at_integration_scale = block_ptr->blockDataAtScale(integration_scale);

    const Eigen::Vector3i voxel_coord_base = block_ptr->coord;
    Eigen::Vector3f sample_point_base_W;
    map_.voxelToPoint(voxel_coord_base, integration_stride, sample_point_base_W);
    const Eigen::Vector3f sample_point_base_C = T_CW_ * sample_point_base_W;
    const int colour_width = has_colour_ ? colour_img_->width() : 0;
    const int colour_height = has_colour_ ? colour_img_->height() : 0;
    const float max_weight = data_config.field.max_weight;

    const Eigen::Matrix3f sample_point_delta_matrix_C =
        (T_CW_.linear()
         * (map_res_
            * (Eigen::Matrix3f() << integration_stride,
               0,
               0,
               0,
               integration_stride,
               0,
               0,
               0,
               integration_stride)
                  .finished()));

    for (unsigned int z = 0; z < size_at_integration_scale_li; z++) {
        for (unsigned int y = 0; y < size_at_integration_scale_li; y++) {
            for (unsigned int x = 0; x < size_at_integration_scale_li; x++) {
                const Eigen::Vector3f sample_point_C =
                    sample_point_base_C + sample_point_delta_matrix_C * Eigen::Vector3f(x, y, z);

                Eigen::Vector2f depth_pixel_f;
                if (sensor_.model.project(sample_point_C, &depth_pixel_f)
                    != srl::projection::ProjectionStatus::Successful) {
                    continue;
                }

                const Eigen::Vector2i depth_pixel = se::round_pixel(depth_pixel_f);
                const float depth_value = depth_img_(depth_pixel.x(), depth_pixel.y());
                if (depth_value < sensor_.near_plane) {
                    continue;
                }

                const float three_sigma = three_sigma_from_depth(depth_value);

                const int voxel_idx =
                    x + y * size_at_integration_scale_li + z * size_at_integration_scale_sq;
                auto& voxel_data = data_at_integration_scale[voxel_idx];

                const float sample_point_C_m = sensor_.measurementFromPoint(sample_point_C);
                const float range = sample_point_C.norm();
                const float range_diff =
                    (sample_point_C_m - depth_value) * (range / sample_point_C_m);
                block_ptr->incrCurrObservedCount(updater::update_voxel(
                    voxel_data, range_diff, tau, three_sigma, data_config));
                const bool field_updated = range_diff < tau;

                Eigen::Vector3f hit_C;
                if constexpr (ColB == Colour::On) {
                    if (has_colour_ && field_updated) {
                        sensor_.model.backProject(depth_pixel_f, &hit_C);
                        hit_C.array() *= depth_value;
                    }
                }

                if constexpr (ColB == Colour::On) {
                    if (has_colour_ && field_updated) {
                        const Eigen::Vector3f hit_Cc = T_CcC_ * hit_C;
                        Eigen::Vector2f colour_pixel_f;
                        if (colour_sensor_->model.project(hit_Cc, &colour_pixel_f)
                            == srl::projection::ProjectionStatus::Successful) {
                            const int u = static_cast<int>(std::lround(
                                (colour_pixel_f.x() + 0.5f) * colour_scale_u_ - 0.5f));
                            const int v = static_cast<int>(std::lround(
                                (colour_pixel_f.y() + 0.5f) * colour_scale_v_ - 0.5f));
                            if (u >= 0 && u < colour_width && v >= 0 && v < colour_height) {
                                voxel_data.colour.update(
                                    (*colour_img_)(u, v), max_weight);
                            }
                        }
                    }
                }
            }
        }
    }

    block_ptr->incrCurrIntegrCount();
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>,
               SensorT>::freeNodeRecurse(OctantBase* octant_ptr,
                                         int depth,
                                         std::vector<std::vector<OctantBase*>>& node_set_local,
                                         std::vector<OctantBase*>& freed_blocks_local)
{
    assert(octant_ptr);
    assert(!octant_ptr->is_block);
    assert(depth >= 0);

    NodeType* node_ptr = static_cast<NodeType*>(octant_ptr);

    if (node_ptr->isLeaf()) {
        typename NodeType::DataType node_data = node_ptr->data();
        node_data.field.update(map_.getDataConfig().field.log_odd_min,
                               map_.getDataConfig().field.max_weight);
        node_ptr->min_data = node_data;
        node_ptr->max_data = node_ptr->min_data;

        node_set_local[depth - 1].push_back(node_ptr->parent());
    }
    else {
        for (int child_idx = 0; child_idx < 8; child_idx++) {
            OctantBase* child_ptr = node_ptr->getChild(child_idx);
            assert(child_ptr);
            if (child_ptr->is_block) {
                freeBlock(child_ptr);
                node_set_local[depth].push_back(child_ptr->parent());
                freed_blocks_local.push_back(child_ptr);
            }
            else {
                freeNodeRecurse(child_ptr, depth + 1, node_set_local, freed_blocks_local);
            }
        }
    }
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>,
               SensorT>::syncCameraCpuEntry(const size_t sample_index,
                                            const gs::Camera& camera)
{
    if (!gs_cam_list_cpu_ || sample_index >= gs_cam_list_cpu_->size()) {
        return;
    }
    auto& cam_cpu = (*gs_cam_list_cpu_)[sample_index];
    cam_cpu.width = camera.width;
    cam_cpu.height = camera.height;
    cam_cpu.fov_x = camera.fov_x;
    cam_cpu.fov_y = camera.fov_y;
    cam_cpu.fx = camera.fx;
    cam_cpu.fy = camera.fy;
    cam_cpu.cx = camera.cx;
    cam_cpu.cy = camera.cy;

    auto T_W2C_cpu = camera.T_W2C.to(torch::kCPU, false).contiguous();
    torch::Tensor proj_cpu;
    if (cam_cpu.proj_matrix_raw.defined()) {

        proj_cpu = cam_cpu.proj_matrix_raw;
    } else if (camera.proj_matrix_raw.defined()) {
        proj_cpu = camera.proj_matrix_raw.to(torch::kCPU, false).contiguous();
    } else {
        proj_cpu = T_W2C_cpu.inverse()
                       .mm(camera.full_proj_matrix.to(torch::kCPU, false).contiguous())
                       .contiguous();
    }
    gs::refreshFixedPoseCameraDerivedTensors(cam_cpu, T_W2C_cpu, proj_cpu);
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>,
               SensorT>::recordJointPoseUpdate(const size_t sample_index,
                                               const bool is_current_frame,
                                               const gs::Camera& camera)
{
    if (!pose_updates_) {
        return;
    }
    gs::PoseUpdate update;
    update.sample_index = sample_index;
    update.is_current_frame = is_current_frame;
    if (extractCameraPoseWorld(camera, &update.T_WC)) {
        pose_updates_->push_back(update);
    }
}

template<Colour ColB, int BlockSize, typename SensorT>
void GSUpdater<Map<Data<Field::Occupancy, ColB>, Res::Multi, BlockSize>, SensorT>::updateGSModel(
    std::vector<gs::Point>& positions,
    std::vector<gs::Color>& colors,
    std::vector<gs::Point>& scales_xyz,
    std::vector<gs::Quat>& rotations_wxyz)
{
    GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/model_update");
    const auto& densify_cfg = gs_model_.densify_params;
    const bool do_densify =
        densify_cfg.enabled && densify_cfg.interval > 0 && (timestamp_ % densify_cfg.interval == 0);

    auto finalizeDataPacketAndPush = [&]() {
        if (!preview_packet_) {
            return;
        }
        end_time_ = PerfStats::timeNow();
        data_packet_.fps = 1.0 / (end_time_ - start_time_);
        data_packet_.ID = timestamp_;
        data_packet_.num_splats = gs_model_.Get_size();
        data_packet_.num_kf = gt_img_list_.size();
        *preview_packet_ = std::move(data_packet_);
        data_packet_ = gs::DataPacket();
    };

    if (!positions.empty()) {
        torch::NoGradGuard no_grad;
        gs_model_.Add_gaussians_oriented(positions, colors, scales_xyz, rotations_wxyz);
    }

    const auto model_xyz = gs_model_.Get_xyz();
    const bool xyz_defined = model_xyz.defined();
    const size_t model_size = xyz_defined ? static_cast<size_t>(model_xyz.size(0)) : 0;
    const bool optimizer_defined = static_cast<bool>(gs_model_.optimizer);
    if (!xyz_defined || model_size == 0 || !optimizer_defined) {
        if (!g_logged_empty_gs_model_skip.exchange(true)) {
            const std::string submap_tag = gs_model_.output_path.filename().string();
            const size_t valid_count = positions.size();
            LOG(WARNING) << "Skipping online GS update for empty or incomplete model."
                         << " submap_tag=" << submap_tag
                         << " frame_ts=" << timestamp_
                         << " positions_size=" << positions.size()
                         << " valid_count=" << valid_count
                         << " xyz_defined=" << xyz_defined
                         << " model_size=" << model_size
                         << " optimizer_defined=" << optimizer_defined
                         << " gs_cam_list_size=" << gs_cam_list_.size()
                         << " gt_img_list_size=" << gt_img_list_.size();
        }
        finalizeDataPacketAndPush();
        return;
    }

    int iters = gs_model_.optimParams.kf_iters;
    if (!isKeyframe_) {
        iters = gs_model_.optimParams.non_kf_iters;
    }

    std::vector<int> kf_indices = gs::get_random_indices(gt_img_list_.size());
    const int current_joint_tail = isKeyframe_
                                       ? resolveJointTailCount(
                                             iters,
                                             gs_model_.optimParams.online_kf_joint_last_iters)
                                       : resolveJointTailCount(
                                             iters,
                                             gs_model_.optimParams.online_non_kf_joint_last_iters);
    const int random_joint_tail = resolveJointTailCount(
        gs_model_.optimParams.random_kf_num,
        gs_model_.optimParams.online_random_kf_joint_last_iters);
    const double pose_rot_lr =
        std::max(0.0, static_cast<double>(gs_model_.optimParams.online_pose_rot_lr));
    const double pose_trans_lr =
        std::max(0.0, static_cast<double>(gs_model_.optimParams.online_pose_trans_lr));

    torch::Tensor densify_valid_mask;
    if (do_densify) {
        const int W = colour_img_->width();
        const int H = colour_img_->height();
        const int depthW = depth_img_.width();
        const int depthH = depth_img_.height();
        const float* depth_ptr = depth_img_.data();
        const float near_depth = sensor_.near_plane;
        const float far_depth = sensor_.far_plane;
        if (W > 0 && H > 0 && depthW > 0 && depthH > 0 && depth_ptr) {
            std::vector<uint8_t> depth_valid_mask_cpu(
                static_cast<size_t>(depthW) * static_cast<size_t>(depthH), uint8_t(0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int v = 0; v < depthH; ++v) {
                const size_t depth_row = static_cast<size_t>(v) * static_cast<size_t>(depthW);
                for (int u = 0; u < depthW; ++u) {
                    const float d = depth_ptr[depth_row + static_cast<size_t>(u)];
                    depth_valid_mask_cpu[depth_row + static_cast<size_t>(u)] =
                        static_cast<uint8_t>(std::isfinite(d) && d > near_depth && d < far_depth);
                }
            }

            const std::vector<int> u_depth_lut = build_colour_to_depth_index_lut(W, depthW);
            const std::vector<int> v_depth_lut = build_colour_to_depth_index_lut(H, depthH);
            std::vector<uint8_t> valid_mask_cpu(
                static_cast<size_t>(W) * static_cast<size_t>(H), uint8_t(0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int v = 0; v < H; ++v) {
                const int v_depth = v_depth_lut[static_cast<size_t>(v)];
                const size_t depth_row = static_cast<size_t>(v_depth) * static_cast<size_t>(depthW);
                const size_t colour_row = static_cast<size_t>(v) * static_cast<size_t>(W);
                for (int u = 0; u < W; ++u) {
                    const int u_depth = u_depth_lut[static_cast<size_t>(u)];
                    valid_mask_cpu[colour_row + static_cast<size_t>(u)] =
                        depth_valid_mask_cpu[depth_row + static_cast<size_t>(u_depth)];
                }
            }

            densify_valid_mask =
                torch::from_blob(valid_mask_cpu.data(),
                                 {static_cast<int64_t>(H), static_cast<int64_t>(W)},
                                 torch::TensorOptions().dtype(torch::kUInt8))
                    .clone()
                    .to(cur_gt_img_.device(), true);
        }
    }

    const float colour_fx = colour_fx_;
    const float colour_fy = colour_fy_;
    const float colour_cx = colour_cx_;
    const float colour_cy = colour_cy_;
    const bool geometry_supervision =
        gs_model_.optimParams.geometrySupervisionActive();

    const bool current_in_list = isKeyframe_ || gs_model_.optimParams.keep_all_frames;
    const size_t current_index =
        (current_in_list && !gs_cam_list_.empty()) ? (gs_cam_list_.size() - 1)
                                                   : std::numeric_limits<size_t>::max();
    const bool enable_joint_current =
        (current_joint_tail > 0) && (pose_rot_lr > 0.0 || pose_trans_lr > 0.0);
    std::unique_ptr<GsPoseOptimState> current_pose_state;
    torch::Tensor current_proj_raw;
    if (enable_joint_current) {
        current_pose_state = std::make_unique<GsPoseOptimState>();
        current_pose_state->theta =
            torch::zeros({3},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                .set_requires_grad(true);
        current_pose_state->rho =
            torch::zeros({3},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                .set_requires_grad(true);
        std::vector<torch::optim::OptimizerParamGroup> pose_params;
        pose_params.reserve(2);
        pose_params.push_back(torch::optim::OptimizerParamGroup(
            {current_pose_state->theta},
            std::make_unique<torch::optim::AdamOptions>(pose_rot_lr)));
        pose_params.push_back(torch::optim::OptimizerParamGroup(
            {current_pose_state->rho},
            std::make_unique<torch::optim::AdamOptions>(pose_trans_lr)));
        current_pose_state->optimizer = std::make_unique<torch::optim::Adam>(
            pose_params, torch::optim::AdamOptions(0.0).eps(1e-15));
        current_proj_raw = cur_gs_cam_.T_W2C.inverse().mm(cur_gs_cam_.full_proj_matrix).contiguous();
    }
    bool current_pose_changed = false;
    auto append_fisher_sample_from_capture =
        [&](torch::Tensor fisher_xyz_diag,
            torch::Tensor fisher_opacity_diag,
            const bool is_current_frame,
            const bool contributes_to_history) {
            if (!fisher_training_samples_) {
                return;
            }
            if ((!fisher_xyz_diag.defined() || fisher_xyz_diag.numel() == 0)
                && (!fisher_opacity_diag.defined() || fisher_opacity_diag.numel() == 0)) {
                return;
            }

            gs::FisherTrainingSample sample;
            sample.is_current_frame = is_current_frame;
            sample.contributes_to_history = contributes_to_history;
            if (fisher_xyz_diag.defined() && fisher_xyz_diag.numel() > 0) {
                sample.diag.xyz = std::move(fisher_xyz_diag);
            }
            if (fisher_opacity_diag.defined() && fisher_opacity_diag.numel() > 0) {
                sample.diag.opacity = std::move(fisher_opacity_diag);
            }
            fisher_training_samples_->push_back(std::move(sample));
        };

    gs::profiling::NvtxRange current_optimization_range(
        "GOFusion/GSUpdater/current_frame_optimization");
    for (int iter = 0; iter < iters; iter++) {
        GOFUSION_NVTX_RANGE("GOFusion/GSUpdater/current_iteration");
        const bool optimize_pose = enable_joint_current && (iter >= iters - current_joint_tail);
        const bool capture_current_fisher =
            fisher_training_samples_ && current_in_list && (iter == iters - 1);
        torch::Tensor image;
        torch::Tensor viewspace_point_tensor;
        torch::Tensor visibility_filter;
        torch::Tensor radii;
        gs::RenderResult geometry_render;
        gs::MonoGaussianBackwardFisherCapture current_fisher_capture;
        if (capture_current_fisher) {
            current_fisher_capture.means3D_diag = torch::zeros_like(gs_model_.Get_xyz());
            current_fisher_capture.opacity_diag = torch::zeros_like(gs_model_.Get_opacity_raw());
        }
        const bool force_mono_render = (do_densify || capture_current_fisher) && (iter == iters - 1);
        gs::profiling::NvtxRange render_range("GOFusion/GSUpdater/render");
        if (geometry_supervision) {
            if (optimize_pose) {
                current_pose_state->optimizer->zero_grad(true);
                geometry_render = gs::render_mono_geometry(
                    cur_gs_cam_,
                    gs_model_,
                    current_pose_state->theta,
                    current_pose_state->rho,
                    1.0f,
                    torch::Tensor(),
                    capture_current_fisher ? &current_fisher_capture : nullptr);
            }
            else if (force_mono_render) {
                geometry_render = gs::render_mono_geometry(
                    cur_gs_cam_,
                    gs_model_,
                    torch::Tensor(),
                    torch::Tensor(),
                    1.0f,
                    torch::Tensor(),
                    capture_current_fisher ? &current_fisher_capture : nullptr);
            }
            else {
                geometry_render = gs::render_faster_geometry(cur_gs_cam_, gs_model_);
            }
            image = geometry_render.rgb;
            viewspace_point_tensor = geometry_render.means2D;
            visibility_filter = geometry_render.visibility;
            radii = geometry_render.radii;
        }
        else if (optimize_pose) {
            current_pose_state->optimizer->zero_grad(true);
            std::tie(image, viewspace_point_tensor, visibility_filter, radii) =
                gs::render_mono_with_pose(cur_gs_cam_,
                                          gs_model_,
                                          current_pose_state->theta,
                                          current_pose_state->rho,
                                          1.0f,
                                          torch::Tensor(),
                                          capture_current_fisher ? &current_fisher_capture
                                                                 : nullptr);
        } else if (force_mono_render) {
            std::tie(image, viewspace_point_tensor, visibility_filter, radii) =
                gs::render_mono(cur_gs_cam_,
                                gs_model_,
                                1.0f,
                                torch::Tensor(),
                                capture_current_fisher ? &current_fisher_capture : nullptr);
        } else {
            std::tie(image, viewspace_point_tensor, visibility_filter, radii) =
                gs::render(cur_gs_cam_, gs_model_);
        }
        render_range.stop();

        gs::profiling::NvtxRange loss_range("GOFusion/GSUpdater/loss");
        auto loss = gs::l1_loss(image, cur_gt_img_);
        if (geometry_supervision) {
            const auto geometry_loss = gs::geometrySupervisionLoss(
                geometry_render,
                cur_geometry_target_,
                gs_model_.optimParams.lambda_depth,
                gs_model_.optimParams.lambda_normal);
            loss = loss + geometry_loss.total;
        }

        gs::DensifyMask dens;
        if (do_densify && iter == iters - 1) {
            dens = gs::build_densify_mask(
                image.detach(), cur_gt_img_, densify_valid_mask, densify_cfg);
            auto region_loss = gs::densify_region_mse(image, cur_gt_img_, dens.mask);
            if (region_loss.defined()) {
                loss = loss + region_loss;
            }
        }
        loss_range.stop();

        gs::profiling::NvtxRange backward_range("GOFusion/GSUpdater/loss_backward");
        loss.backward();
        backward_range.stop();
        if (capture_current_fisher) {
            append_fisher_sample_from_capture(std::move(current_fisher_capture.means3D_diag),
                                              std::move(current_fisher_capture.opacity_diag),
                                              true,
                                              true);
        }
        gs::profiling::NvtxRange optimizer_range("GOFusion/GSUpdater/optimizer_step");
        gs::fused_adam_step(gs_model_.optimizer.get());
        if (optimize_pose) {
            current_pose_state->optimizer->step();
            applyPoseDeltaToCamera(
                cur_gs_cam_, current_pose_state->rho, current_pose_state->theta, current_proj_raw);
            current_pose_state->theta.data().zero_();
            current_pose_state->rho.data().zero_();
            current_pose_state->optimizer->zero_grad(true);
            current_pose_changed = true;
        }
        optimizer_range.stop();

        gs_model_.optimizer->zero_grad(true);


        if (do_densify && iter == iters - 1) {
            torch::NoGradGuard no_grad;
            const int W = colour_img_->width();
            const int H = colour_img_->height();
            const int depthW = depth_img_.width();
            const int depthH = depth_img_.height();
            const float colour_to_depth_scale_x =
                static_cast<float>(depthW) / static_cast<float>(W);
            const float colour_to_depth_scale_y =
                static_cast<float>(depthH) / static_cast<float>(H);
            const float* depth_ptr = depth_img_.data();
            const float depth_fx = sensor_.model.focalLengthU();
            const float depth_fy = sensor_.model.focalLengthV();
            const float depth_cx = sensor_.model.imageCenterU();
            const float depth_cy = sensor_.model.imageCenterV();

            const Eigen::Matrix3f R_WCc = T_WCc_.linear();
            const Eigen::Vector3f t_WCc = T_WCc_.translation();
            const Eigen::Vector3f exW = R_WCc.col(0);
            const Eigen::Vector3f eyW = R_WCc.col(1);

            const bool have_leaf =
                (g_qtree_leaf_W == W && g_qtree_leaf_H == H
                 && g_qtree_leaf_id.size() == static_cast<size_t>(W) * static_cast<size_t>(H)
                 && g_qtree_leaves.size() == static_cast<size_t>(g_qtree_leaf_count)
                 && g_qtree_leaf_count > 0);

            const double peak_ratio = std::clamp<double>(densify_cfg.top_peaks_ratio, 0.0, 1.0);

            if (have_leaf && peak_ratio > 0.0
                && dens.peak_x.defined() && dens.peak_y.defined()
                && dens.peak_x.numel() > 0 && dens.peak_y.numel() == dens.peak_x.numel()
                && densify_cfg.max_new_gaussians > 0) {

                std::vector<int64_t> leaf_best_gid(static_cast<size_t>(g_qtree_leaf_count), -1);
                std::vector<float>   leaf_best_depth(static_cast<size_t>(g_qtree_leaf_count), 1.0e30f);
                std::vector<float>   leaf_best_maxxy(static_cast<size_t>(g_qtree_leaf_count), 0.f);

                auto vis = visibility_filter.to(torch::kBool);
                auto vis_idx = torch::nonzero(vis).reshape({-1});
                if (vis_idx.defined() && vis_idx.numel() > 0) {
                    auto vis_idx_cpu = vis_idx.to(torch::kCPU).contiguous();
                    // T_W2C stores the transposed world-to-camera transform.
                    using torch::indexing::Slice;
                    const auto& view = cur_gs_cam_.T_W2C;
                    auto xyz_vis = gs_model_.Get_xyz().index_select(0, vis_idx);
                    auto vsp_vis_cpu =
                        (xyz_vis.mm(view.index({Slice(0, 3), Slice(0, 3)}))
                         + view.index({3, Slice(0, 3)}))
                            .to(torch::kCPU).contiguous();
                    auto scl_vis_cpu =
                        gs_model_.Get_scaling().index_select(0, vis_idx).to(torch::kCPU).contiguous();

                    const int64_t Nv = vis_idx_cpu.numel();
                    const int64_t* gid_ptr = vis_idx_cpu.data_ptr<int64_t>();
                    const float* vsp_ptr = vsp_vis_cpu.data_ptr<float>();
                    const float* scl_ptr = scl_vis_cpu.data_ptr<float>();

                    for (int64_t j = 0; j < Nv; ++j) {
                        const float x = vsp_ptr[j * 3 + 0];
                        const float y = vsp_ptr[j * 3 + 1];
                        const float z = vsp_ptr[j * 3 + 2];
                        if (!(z > 1.0e-6f)) {
                            continue;
                        }

                        const int u = static_cast<int>(std::lround(double(colour_fx * x / z + colour_cx)));
                        const int v = static_cast<int>(std::lround(double(colour_fy * y / z + colour_cy)));
                        if (u < 0 || u >= W || v < 0 || v >= H) {
                            continue;
                        }

                        const int pidx = u + W * v;
                        const int leaf = g_qtree_leaf_id[static_cast<size_t>(pidx)];
                        if (leaf < 0 || leaf >= g_qtree_leaf_count) {
                            continue;
                        }

                        const float mx = scl_ptr[j * 3 + 0];
                        const float my = scl_ptr[j * 3 + 1];
                        const float maxxy = (mx > my) ? mx : my;

                        const size_t lid = static_cast<size_t>(leaf);
                        if (z < leaf_best_depth[lid]) {
                            leaf_best_depth[lid] = z;
                            leaf_best_gid[lid] = gid_ptr[j];
                            leaf_best_maxxy[lid] = maxxy;
                        }
                    }
                }

                auto px_cpu = dens.peak_x.to(torch::kCPU).contiguous();
                auto py_cpu = dens.peak_y.to(torch::kCPU).contiguous();
                const int64_t K = px_cpu.numel();
                int64_t N = static_cast<int64_t>(std::ceil(peak_ratio * static_cast<double>(K)));
                if (N < 1) N = 1;
                if (N > K) N = K;

                int budget = densify_cfg.max_new_gaussians;

                std::vector<gs::Point> add_pos;
                std::vector<gs::Color> add_col;
                std::vector<gs::Point> add_sxyz;
                std::vector<gs::Quat>  add_rot;

                add_pos.reserve(static_cast<size_t>(std::min<int64_t>(K, budget)));
                add_col.reserve(add_pos.capacity());
                add_sxyz.reserve(add_pos.capacity());
                add_rot.reserve(add_pos.capacity());

                const int64_t* px = px_cpu.data_ptr<int64_t>();
                const int64_t* py = py_cpu.data_ptr<int64_t>();

                for (int64_t i = 0; i < N && budget > 0; ++i) {
                    const int u = static_cast<int>(px[i]);
                    const int v = static_cast<int>(py[i]);
                    if (u < 0 || u >= W || v < 0 || v >= H) {
                        continue;
                    }
                    const int pidx = u + W * v;

                    const int leaf = g_qtree_leaf_id[static_cast<size_t>(pidx)];
                    const gs::Leaf* qtree_leaf =
                        (leaf >= 0 && leaf < g_qtree_leaf_count)
                            ? &g_qtree_leaves[static_cast<size_t>(leaf)]
                            : nullptr;
                    const float lw_depth_px =
                        static_cast<float>(qtree_leaf ? qtree_leaf->getWidth()
                                                       : densify_cfg.tile_size)
                        * colour_to_depth_scale_x;
                    const float lh_depth_px =
                        static_cast<float>(qtree_leaf ? qtree_leaf->getHeight()
                                                       : densify_cfg.tile_size)
                        * colour_to_depth_scale_y;

                    const int u_depth = g_colour_depth_index_cache.identity
                                            ? u
                                            : g_colour_depth_index_cache.u[static_cast<size_t>(u)];
                    const int v_depth = g_colour_depth_index_cache.identity
                                            ? v
                                            : g_colour_depth_index_cache.v[static_cast<size_t>(v)];

                    const float d = depth_ptr[u_depth + depthW * v_depth];
                    if (!(std::isfinite(d) && d > sensor_.near_plane && d < sensor_.far_plane)) {
                        continue;
                    }

                    gs::OrientedGaussianInit init;
                    if (!gs::make_oriented_gaussian_from_depth_pixel_precomp(
                            &init,
                            u_depth,
                            v_depth,
                            lw_depth_px,
                            lh_depth_px,
                            depth_ptr,
                            depthW,
                            depthH,
                            sensor_.near_plane,
                            depth_fx,
                            depth_fy,
                            depth_cx,
                            depth_cy,
                            R_WCc,
                            t_WCc,
                            exW,
                            eyW,
                            densify_cfg.init_scale_mul,
                            densify_cfg.init_z_scale)) {
                        continue;
                    }

                    const float target_max =
                        (init.scale_xyz.x > init.scale_xyz.y) ? init.scale_xyz.x : init.scale_xyz.y;

                    int64_t covered_gid = -1;
                    float covered_maxxy = 0.f;
                    float covered_depth = 1.0e30f;

                    if (leaf >= 0 && leaf < g_qtree_leaf_count) {
                        const size_t lid = static_cast<size_t>(leaf);
                        covered_gid = leaf_best_gid[lid];
                        covered_maxxy = leaf_best_maxxy[lid];
                        covered_depth = leaf_best_depth[lid];

                        const float depth_eps = std::max(2.0f * target_max, 0.02f * d);
                        if (std::abs(covered_depth - d) > depth_eps) {
                            covered_gid = -1;
                        }
                    }

                    auto rgb = (*colour_img_)[u + W * v];
                    gs::Color c;
                    c.r = rgb.r;
                    c.g = rgb.g;
                    c.b = rgb.b;

                    const bool do_split =
                        (covered_gid >= 0)
                        && (densify_cfg.split_factor >= 2)
                        && (covered_maxxy > densify_cfg.split_size_ratio * target_max)
                        && (budget >= densify_cfg.split_factor);

                    if (do_split) {
                        const int M = densify_cfg.split_factor;

                        const float qw = init.rot_wxyz.w;
                        const float qx = init.rot_wxyz.x;
                        const float qy = init.rot_wxyz.y;
                        const float qz = init.rot_wxyz.z;
                        const Eigen::Vector3f t1(1.f - 2.f * (qy * qy + qz * qz),
                                                 2.f * (qx * qy + qz * qw),
                                                 2.f * (qx * qz - qy * qw));

                        for (int s = 0; s < M; ++s) {
                            const float a = (M == 1) ? 0.f : (-0.5f + float(s) / float(M - 1));
                            const float off = a * target_max;

                            gs::Point p;
                            p.x = init.pos_w.x + t1.x() * off;
                            p.y = init.pos_w.y + t1.y() * off;
                            p.z = init.pos_w.z + t1.z() * off;

                            gs::Point sc = init.scale_xyz;
                            sc.x *= densify_cfg.split_scale_shrink;
                            sc.y *= densify_cfg.split_scale_shrink;
                            sc.z = densify_cfg.init_z_scale;

                            add_pos.push_back(p);
                            add_col.push_back(c);
                            add_sxyz.push_back(sc);
                            add_rot.push_back(init.rot_wxyz);
                        }
                        budget -= M;
                    }
                    else {
                        add_pos.push_back(init.pos_w);
                        add_col.push_back(c);
                        add_sxyz.push_back(init.scale_xyz);
                        add_rot.push_back(init.rot_wxyz);
                        budget -= 1;
                    }
                }

                if (!add_pos.empty()) {
                    gs_model_.Add_gaussians_oriented(add_pos, add_col, add_sxyz, add_rot);
                }
            }
        }

        if (preview_packet_ && iter == iters - 1) {
            try {
                auto rendered_rgb_hwc =
                    image.detach().permute({1, 2, 0}).contiguous();
                data_packet_.rendered_rgb_staging =
                    gs::stagePreviewRgbHwcFloatToCpu(rendered_rgb_hwc);
            }
            catch (const std::exception& error) {
                LOG(WARNING) << "GS rendered preview staging dropped a frame: "
                             << error.what();
            }
        }
    }
    current_optimization_range.stop();

    if (current_pose_changed) {
        if (current_index != std::numeric_limits<size_t>::max() && current_index < gs_cam_list_.size()) {
            gs_cam_list_[current_index] = cur_gs_cam_;
            syncCameraCpuEntry(current_index, cur_gs_cam_);
            recordJointPoseUpdate(current_index, true, cur_gs_cam_);
        } else {
            recordJointPoseUpdate(std::numeric_limits<size_t>::max(), true, cur_gs_cam_);
        }
    }

    if (!isKeyframe_) {
        gs::profiling::NvtxRange random_kf_range(
            "GOFusion/GSUpdater/random_keyframe_optimization");
        int kf_iters = gs_model_.optimParams.random_kf_num;
        if (static_cast<int>(kf_indices.size()) < kf_iters) {
            kf_iters = static_cast<int>(kf_indices.size());
        }
        for (int i = 0; i < kf_iters; i++) {
            const int sample_index = kf_indices[i];
            auto kf_gt_img = gt_img_list_[sample_index];
            auto& kf_gs_cam = gs_cam_list_[sample_index];
            const bool optimize_pose = (pose_rot_lr > 0.0 || pose_trans_lr > 0.0)
                                       && (i >= kf_iters - random_joint_tail);
            const bool capture_kf_fisher = fisher_training_samples_ != nullptr;
            gs::MonoGaussianBackwardFisherCapture fisher_capture;
            if (capture_kf_fisher) {
                fisher_capture.means3D_diag = torch::zeros_like(gs_model_.Get_xyz());
                fisher_capture.opacity_diag = torch::zeros_like(gs_model_.Get_opacity_raw());
            }

            torch::Tensor image;
            gs::RenderResult geometry_render;
            if (geometry_supervision
                && static_cast<size_t>(sample_index) >= geometry_target_list_.size()) {
                throw std::runtime_error("Random keyframe geometry target index mismatch");
            }
            if (optimize_pose) {
                GsPoseOptimState pose_state;
                pose_state.theta =
                    torch::zeros({3},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                        .set_requires_grad(true);
                pose_state.rho =
                    torch::zeros({3},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                        .set_requires_grad(true);
                std::vector<torch::optim::OptimizerParamGroup> pose_params;
                pose_params.reserve(2);
                pose_params.push_back(torch::optim::OptimizerParamGroup(
                    {pose_state.theta},
                    std::make_unique<torch::optim::AdamOptions>(pose_rot_lr)));
                pose_params.push_back(torch::optim::OptimizerParamGroup(
                    {pose_state.rho},
                    std::make_unique<torch::optim::AdamOptions>(pose_trans_lr)));
                pose_state.optimizer = std::make_unique<torch::optim::Adam>(
                    pose_params, torch::optim::AdamOptions(0.0).eps(1e-15));
                const torch::Tensor proj_raw =
                    kf_gs_cam.T_W2C.inverse().mm(kf_gs_cam.full_proj_matrix).contiguous();
                pose_state.optimizer->zero_grad(true);
                if (geometry_supervision) {
                    geometry_render = gs::render_mono_geometry(
                        kf_gs_cam,
                        gs_model_,
                        pose_state.theta,
                        pose_state.rho,
                        1.0f,
                        torch::Tensor(),
                        capture_kf_fisher ? &fisher_capture : nullptr);
                    image = geometry_render.rgb;
                }
                else {
                    std::tie(image, std::ignore, std::ignore, std::ignore) =
                        gs::render_mono_with_pose(
                            kf_gs_cam,
                            gs_model_,
                            pose_state.theta,
                            pose_state.rho,
                            1.0f,
                            torch::Tensor(),
                            capture_kf_fisher ? &fisher_capture : nullptr);
                }

                auto loss = gs::l1_loss(image, kf_gt_img);
                if (geometry_supervision) {
                    loss = loss + gs::geometrySupervisionLoss(
                                      geometry_render,
                                      geometry_target_list_[sample_index],
                                      gs_model_.optimParams.lambda_depth,
                                      gs_model_.optimParams.lambda_normal)
                                      .total;
                }
                loss.backward();
                append_fisher_sample_from_capture(std::move(fisher_capture.means3D_diag),
                                                  std::move(fisher_capture.opacity_diag),
                                                  false,
                                                  true);
                gs::fused_adam_step(gs_model_.optimizer.get());
                gs_model_.optimizer->zero_grad(true);
                pose_state.optimizer->step();
                applyPoseDeltaToCamera(kf_gs_cam, pose_state.rho, pose_state.theta, proj_raw);
                pose_state.theta.data().zero_();
                pose_state.rho.data().zero_();
                pose_state.optimizer->zero_grad(true);
                syncCameraCpuEntry(static_cast<size_t>(sample_index), kf_gs_cam);
                recordJointPoseUpdate(static_cast<size_t>(sample_index), false, kf_gs_cam);
                continue;
            }

            if (geometry_supervision) {
                geometry_render = gs::render_mono_geometry(
                    kf_gs_cam,
                    gs_model_,
                    torch::Tensor(),
                    torch::Tensor(),
                    1.0f,
                    torch::Tensor(),
                    capture_kf_fisher ? &fisher_capture : nullptr);
                image = geometry_render.rgb;
            }
            else {
                std::tie(image, std::ignore, std::ignore, std::ignore) =
                    gs::render_mono(kf_gs_cam,
                                    gs_model_,
                                    1.0f,
                                    torch::Tensor(),
                                    capture_kf_fisher ? &fisher_capture : nullptr);
            }
            auto loss = gs::l1_loss(image, kf_gt_img);
            if (geometry_supervision) {
                loss = loss + gs::geometrySupervisionLoss(
                                  geometry_render,
                                  geometry_target_list_[sample_index],
                                  gs_model_.optimParams.lambda_depth,
                                  gs_model_.optimParams.lambda_normal)
                                  .total;
            }
            loss.backward();
            append_fisher_sample_from_capture(std::move(fisher_capture.means3D_diag),
                                              std::move(fisher_capture.opacity_diag),
                                              false,
                                              true);
            gs::fused_adam_step(gs_model_.optimizer.get());
            gs_model_.optimizer->zero_grad(true);
        }

    }

    finalizeDataPacketAndPush();
}

} 

#endif 
