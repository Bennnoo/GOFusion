/*
 * SPDX-FileCopyrightText: 2023 Janusch Patas
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * All rights reserved. Some of the code is derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#ifndef GS_GAUSSIAN_HPP
#define GS_GAUSSIAN_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <torch/torch.h>

#include "gaussian_utils.cuh"
#include "gs/error_driven_densify_params.hpp"

namespace gs {

namespace param {
struct OptimizationParameters {
    int sh_degree = 3;
    float position_lr = 0.00016f;
    float feature_lr = 0.0025f;
    float opacity_lr = 0.05f;
    float scaling_lr = 0.001f;
    float rotation_lr = 0.001f;
    float lambda_dssim = 0.2f;

    bool geometry_supervision_enabled = false;
    bool require_input_depth = true;
    float lambda_depth = 0.0f;
    float lambda_normal = 0.0f;

    float qtree_thresh = 0.1f;
    int qtree_min_pixel_size = 1;

    float qtree_ring_thresh = 0.005f;
    int qtree_ring_t = 1;

    int kf_thresh = 50;
    int kf_iters = 10;
    int non_kf_iters = 5;
    int random_kf_num = 5;
    int online_kf_joint_last_iters = 1;
    int online_non_kf_joint_last_iters = 1;
    int online_random_kf_joint_last_iters = 1;
    float online_pose_rot_lr = 0.0003f;
    float online_pose_trans_lr = 0.0001f;
    int global_gaussian_only_iters = 5;
    int global_joint_pose_iters = 0;
    float global_pose_rot_lr = 0.003f;
    float global_pose_trans_lr = 0.001f;

    bool keep_all_frames = false;

    bool geometrySupervisionActive() const
    {
        return geometry_supervision_enabled
            && (lambda_depth > 0.0f || lambda_normal > 0.0f);
    }
};

OptimizationParameters read_optim_params_from_json(const std::string& path);
} 

struct Quat {
    float w;
    float x;
    float y;
    float z;
};

struct PlyTensorsCPU {
    torch::Tensor xyz;      
    torch::Tensor f_dc;     
    torch::Tensor f_rest;   
    torch::Tensor opacity;  
    torch::Tensor scale;    
    torch::Tensor rotation; 
};

;

;

class GaussianModel {
    public:
    explicit GaussianModel(const param::OptimizationParameters& params,
                           const std::string& ply_path);

    GaussianModel(const GaussianModel& other) = delete;

    GaussianModel& operator=(const GaussianModel& other) = delete;

    GaussianModel(GaussianModel&& other) = default;

    GaussianModel& operator=(GaussianModel&& other) = default;

    inline torch::Tensor Get_xyz() const
    {
        return _xyz;
    }
    inline torch::Tensor Get_opacity() const
    {
        return torch::sigmoid(_opacity);
    }
    inline torch::Tensor Get_rotation() const
    {
        return torch::nn::functional::normalize(_rotation);
    }
    inline torch::Tensor Get_scaling()
    {
        return torch::exp(_scaling);
    }
    inline torch::Tensor Get_opacity_raw() const
    {
        return _opacity;
    }
    inline torch::Tensor Get_rotation_raw() const
    {
        return _rotation;
    }
    inline torch::Tensor Get_scaling_raw() const
    {
        return _scaling;
    }
    inline torch::Tensor Get_features_dc_raw() const
    {
        return _features_dc;
    }
    inline torch::Tensor Get_features_rest_raw() const
    {
        return _features_rest;
    }
    inline int Get_active_sh_degree() const
    {
        return _sh_degree;
    }
    inline int Get_size() const
    {
        return _xyz.defined() ? _xyz.size(0) : 0;
    }
    inline int64_t Get_capacity() const
    {
        return _capacity_storage_enabled ? _capacity : Get_size();
    }
    torch::Tensor Get_background_cuda();

    void Add_gaussians(std::vector<Point>& positions,
                       std::vector<Color>& colors,
                       std::vector<float>& scales);

    void Add_gaussians_oriented(std::vector<Point>& positions,
                                std::vector<Color>& colors,
                                std::vector<Point>& scales_xyz,
                                std::vector<Quat>& rotations_wxyz);

    static void Save_ply_tensors_cpu(const PlyTensorsCPU& tensors,
                                     const std::filesystem::path& file_path,
                                     int iteration);
    PlyTensorsCPU Export_ply_tensors_cpu() const;

    void Import_ply_tensors_cpu(const PlyTensorsCPU& t);
    void Release_training_state(bool shrink_capacity = true);

    std::unique_ptr<torch::optim::Adam> optimizer;

    param::OptimizationParameters optimParams;
    gs::ErrorDensifyParams densify_params;
    std::filesystem::path output_path;
    torch::Tensor background = torch::tensor({0.f, 0.f, 0.f});

    private:
    static constexpr size_t kParameterGroupCount = 6;
    using TensorGroup = std::array<torch::Tensor, kParameterGroupCount>;
    using AdamSteps = std::array<int64_t, kParameterGroupCount>;

    void training_setup();
    void initialize_model_tensors(TensorGroup tensors);
    void append_model_tensors(TensorGroup tensors);
    void initialize_capacity_storage(const TensorGroup& tensors);
    void append_capacity_storage(const TensorGroup& tensors);
    bool reserve_capacity(int64_t required_size);

    void shrink_capacity_to_size();
    void bind_active_capacity_views(const AdamSteps& steps);
    AdamSteps capture_adam_steps() const;
    int64_t choose_growth_capacity(int64_t required_size) const;

    uint64_t active_parameter_payload_bytes() const;
    torch::Tensor& parameter_view(size_t index);
    const torch::Tensor& parameter_view(size_t index) const;
    void reset_model_storage();
    void mark_capacity_event(const char* operation,
                             uint64_t copied_bytes,
                             uint64_t avoided_bytes) const;

    int _sh_degree = 0;
    bool _init_status = false;
    torch::Tensor _xyz;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;

    bool _capacity_storage_enabled = true;
    int64_t _size = 0;
    int64_t _capacity = 0;
    TensorGroup _parameter_storage;
    TensorGroup _adam_exp_avg_storage;
    TensorGroup _adam_exp_avg_sq_storage;
    uint64_t _append_count = 0;
    uint64_t _reserve_count = 0;
    uint64_t _payload_bytes_copied = 0;
    uint64_t _payload_bytes_avoided = 0;
    uint64_t _last_payload_bytes_copied = 0;
    uint64_t _last_payload_bytes_avoided = 0;
};

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              int param_position);

void fused_adam_step(torch::optim::Adam* optimizer);

} 

#endif 
