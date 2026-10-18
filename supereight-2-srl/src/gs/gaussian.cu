/*
 * SPDX-FileCopyrightText: 2023 Janusch Patas
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * All rights reserved. Some of the code is derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nlohmann/json.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "gs/gaussian.cuh"
#include "gs/gaussian_utils.cuh"
#include "gs/nvtx_utils.hpp"

using Slice = torch::indexing::Slice;

namespace gs {

namespace {

template<typename OptimizerT>
auto optimizer_state_key(OptimizerT* optimizer, const torch::Tensor& tensor)
{
    using StateMapT = std::decay_t<decltype(optimizer->state())>;
    using StateKeyT = typename StateMapT::key_type;
    if constexpr (std::is_same_v<StateKeyT, void*>) {
        return static_cast<void*>(tensor.unsafeGetTensorImpl());
    }
    else {
        return std::to_string(reinterpret_cast<std::uintptr_t>(tensor.unsafeGetTensorImpl()));
    }
}

bool capacity_storage_enabled_from_environment()
{
    const char* value = std::getenv("GOFUSION_GS_CAPACITY_STORAGE");
    if (!value || value[0] == '\0') {
        return true;
    }
    return std::strcmp(value, "0") != 0
           && std::strcmp(value, "false") != 0
           && std::strcmp(value, "FALSE") != 0
           && std::strcmp(value, "off") != 0
           && std::strcmp(value, "OFF") != 0;
}

bool capacity_trace_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("GOFUSION_GS_CAPACITY_TRACE");
        return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void capacity_trace(const std::string& message)
{
    if (!capacity_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "[GS capacity] %s\n", message.c_str());
    std::fflush(stderr);
}

int64_t round_up_capacity(const int64_t value, const int64_t granularity)
{
    if (value <= 0) {
        return 0;
    }
    return ((value + granularity - 1) / granularity) * granularity;
}
} 

namespace param {
OptimizationParameters read_optim_params_from_json(const std::string& path)
{
    std::filesystem::path json_path = path;

    if (!std::filesystem::exists(json_path)) {
        throw std::runtime_error("Error: " + json_path.string() + " does not exist!");
    }

    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("OptimizationParameter file could not be opened.");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonString = buffer.str();
    file.close();

    nlohmann::json json = nlohmann::json::parse(jsonString);

    OptimizationParameters params;
    params.sh_degree = json["sh_degree"];
    params.position_lr = json["position_lr"];
    params.feature_lr = json["feature_lr"];
    params.opacity_lr = json["opacity_lr"];
    params.scaling_lr = json["scaling_lr"];
    params.rotation_lr = json["rotation_lr"];
    params.lambda_dssim = json["lambda_dssim"];
    if (json.contains("geometry_supervision_enabled")) {
        params.geometry_supervision_enabled = json["geometry_supervision_enabled"];
    }
    if (json.contains("require_input_depth")) {
        params.require_input_depth = json["require_input_depth"];
    }
    if (json.contains("lambda_depth")) {
        params.lambda_depth = std::max(0.0f, json["lambda_depth"].get<float>());
    }
    if (json.contains("lambda_normal")) {
        params.lambda_normal = std::max(0.0f, json["lambda_normal"].get<float>());
    }
    params.qtree_thresh = json["qtree_thresh"];
    params.qtree_min_pixel_size = json["qtree_min_pixel_size"];

    if (json.contains("qtree_ring_thresh")) {
        params.qtree_ring_thresh = json["qtree_ring_thresh"];
    }
    if (json.contains("qtree_ring_t")) {
        params.qtree_ring_t = json["qtree_ring_t"];
    }

    params.kf_thresh = json["kf_thresh"];
    params.kf_iters = json["kf_iters"];
    params.non_kf_iters = json["non_kf_iters"];
    params.random_kf_num = json["random_kf_num"];
    if (json.contains("online_kf_joint_last_iters")) {
        params.online_kf_joint_last_iters = json["online_kf_joint_last_iters"];
    }
    if (json.contains("online_non_kf_joint_last_iters")) {
        params.online_non_kf_joint_last_iters = json["online_non_kf_joint_last_iters"];
    }
    if (json.contains("online_random_kf_joint_last_iters")) {
        params.online_random_kf_joint_last_iters = json["online_random_kf_joint_last_iters"];
    }
    if (json.contains("online_pose_rot_lr")) {
        params.online_pose_rot_lr = json["online_pose_rot_lr"];
    }
    if (json.contains("online_pose_trans_lr")) {
        params.online_pose_trans_lr = json["online_pose_trans_lr"];
    }
    if (json.contains("global_gaussian_only_iters")) {
        params.global_gaussian_only_iters = json["global_gaussian_only_iters"];
    }
    if (json.contains("global_joint_pose_iters")) {
        params.global_joint_pose_iters = json["global_joint_pose_iters"];
    }
    if (json.contains("global_pose_rot_lr")) {
        params.global_pose_rot_lr = json["global_pose_rot_lr"];
    }
    if (json.contains("global_pose_trans_lr")) {
        params.global_pose_trans_lr = json["global_pose_trans_lr"];
    }

    params.keep_all_frames = json["keep_all_frames"];

    if (params.geometrySupervisionActive() && !params.require_input_depth) {
        throw std::runtime_error(
            "Exact GS geometry supervision requires require_input_depth=true");
    }

    return params;
}
} 

GaussianModel::GaussianModel(const param::OptimizationParameters& params,
                             const std::string& ply_path) :
        optimParams(params),
        output_path(ply_path),
        _sh_degree(params.sh_degree),
        _capacity_storage_enabled(capacity_storage_enabled_from_environment())
{
}

torch::Tensor GaussianModel::Get_background_cuda()
{
    if (!background.defined() || background.numel() != 3) {
        background =
            torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    else if (!background.is_cuda() || background.scalar_type() != torch::kFloat32) {
        background = background.to(
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    else if (!background.is_contiguous()) {
        background = background.contiguous();
    }
    return background;
}

void GaussianModel::Add_gaussians(std::vector<Point>& positions,
                                  std::vector<Color>& colors,
                                  std::vector<float>& scales)
{
    if (positions.empty()) {
        return;
    }
    if (positions.size() != colors.size() || positions.size() != scales.size()) {
        throw std::runtime_error("Add_gaussians: input size mismatch");
    }

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    const auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    const long count = static_cast<long>(positions.size());

    auto new_xyz =
        torch::from_blob(positions.data(), {count, 3}, pointType).to(torch::kCUDA);
    auto new_scaling =
        torch::log(torch::from_blob(scales.data(), {count, 1}, pointType))
            .repeat({1, 3})
            .to(torch::kCUDA, true);
    auto new_rotation =
        torch::zeros({count, 4}, torch::TensorOptions().dtype(torch::kFloat32))
            .index_put_({Slice(), 0}, 1)
            .to(torch::kCUDA, true);
    auto new_opacity =
        inverse_sigmoid(0.5 * torch::ones({count, 1})).to(torch::kCUDA, true);

    auto fused_color =
        RGB2SH(torch::from_blob(colors.data(), {count, 3}, colorType).to(pointType) / 255.f)
            .to(torch::kCUDA);
    auto features =
        torch::zeros({count, 3, static_cast<long>(std::pow((_sh_degree + 1), 2))},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    features.index_put_({Slice(), Slice(torch::indexing::None, 3), 0}, fused_color);
    auto new_features_dc =
        features.index({Slice(), Slice(), Slice(0, 1)}).transpose(1, 2).contiguous();
    auto new_features_rest =
        features.index({Slice(), Slice(), Slice(1, torch::indexing::None)})
            .transpose(1, 2)
            .contiguous();

    TensorGroup tensors = {new_xyz,
                           new_features_dc,
                           new_features_rest,
                           new_scaling,
                           new_rotation,
                           new_opacity};
    if (!_init_status) {
        initialize_model_tensors(std::move(tensors));
    }
    else {
        append_model_tensors(std::move(tensors));
    }
}

void GaussianModel::Add_gaussians_oriented(std::vector<Point>& positions,
                                          std::vector<Color>& colors,
                                          std::vector<Point>& scales_xyz,
                                          std::vector<Quat>& rotations_wxyz)
{
    if (positions.empty()) {
        return;
    }
    if (positions.size() != colors.size()
        || positions.size() != scales_xyz.size()
        || positions.size() != rotations_wxyz.size()) {
        throw std::runtime_error("Add_gaussians_oriented: input size mismatch");
    }

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    const auto colorType = torch::TensorOptions().dtype(torch::kUInt8);

    const long N = static_cast<long>(positions.size());

    auto scales3 = torch::from_blob(scales_xyz.data(), {N, 3}, pointType);
    scales3 = scales3.clamp_min(1.0e-6f);
    auto log_scales3 = torch::log(scales3);

    auto rot4 = torch::from_blob(rotations_wxyz.data(), {N, 4}, pointType);

    auto new_xyz = torch::from_blob(positions.data(), {N, 3}, pointType).to(torch::kCUDA);
    auto new_scaling = log_scales3.to(torch::kCUDA, true);
    auto new_rotation = rot4.to(torch::kCUDA, true);
    auto new_opacity =
        inverse_sigmoid(0.5 * torch::ones({N, 1})).to(torch::kCUDA, true);

    auto fused_color =
        RGB2SH(torch::from_blob(colors.data(), {N, 3}, colorType).to(pointType) / 255.f)
            .to(torch::kCUDA);
    auto features =
        torch::zeros({N, 3, static_cast<long>(std::pow((_sh_degree + 1), 2))},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    features.index_put_({Slice(), Slice(torch::indexing::None, 3), 0}, fused_color);
    auto new_features_dc =
        features.index({Slice(), Slice(), Slice(0, 1)}).transpose(1, 2).contiguous();
    auto new_features_rest =
        features.index({Slice(), Slice(), Slice(1, torch::indexing::None)})
            .transpose(1, 2)
            .contiguous();

    TensorGroup tensors = {new_xyz,
                           new_features_dc,
                           new_features_rest,
                           new_scaling,
                           new_rotation,
                           new_opacity};
    if (!_init_status) {
        initialize_model_tensors(std::move(tensors));
    }
    else {
        append_model_tensors(std::move(tensors));
    }
}

torch::Tensor& GaussianModel::parameter_view(const size_t index)
{
    switch (index) {
    case 0:
        return _xyz;
    case 1:
        return _features_dc;
    case 2:
        return _features_rest;
    case 3:
        return _scaling;
    case 4:
        return _rotation;
    case 5:
        return _opacity;
    default:
        throw std::out_of_range("GaussianModel parameter group index");
    }
}

const torch::Tensor& GaussianModel::parameter_view(const size_t index) const
{
    switch (index) {
    case 0:
        return _xyz;
    case 1:
        return _features_dc;
    case 2:
        return _features_rest;
    case 3:
        return _scaling;
    case 4:
        return _rotation;
    case 5:
        return _opacity;
    default:
        throw std::out_of_range("GaussianModel parameter group index");
    }
}

int64_t GaussianModel::choose_growth_capacity(const int64_t required_size) const
{
    constexpr int64_t granularity = 16384;
    const int64_t append_hint =
        std::max<int64_t>(1024, static_cast<int64_t>(densify_params.max_new_gaussians));
    if (_capacity <= 0) {
        return round_up_capacity(std::max(required_size, append_hint), granularity);
    }

    const int64_t geometric_growth = std::max(_capacity / 2, append_hint);
    return round_up_capacity(
        std::max(required_size, _capacity + geometric_growth), granularity);
}

uint64_t GaussianModel::active_parameter_payload_bytes() const
{
    uint64_t bytes = 0;
    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        const auto& tensor = parameter_view(i);
        if (tensor.defined()) {
            bytes += static_cast<uint64_t>(tensor.numel())
                     * static_cast<uint64_t>(tensor.element_size());
        }
    }
    return bytes;
}

void GaussianModel::mark_capacity_event(const char* operation,
                                        const uint64_t copied_bytes,
                                        const uint64_t avoided_bytes) const
{
    if (!capacity_trace_enabled()) {
        return;
    }
    std::ostringstream message;
    message << operation << " size=" << _size << " capacity=" << _capacity
            << " reserves=" << _reserve_count << " copied=" << copied_bytes
            << " avoided=" << avoided_bytes;
    capacity_trace(message.str());
}

GaussianModel::AdamSteps GaussianModel::capture_adam_steps() const
{
    AdamSteps steps{};
    if (!optimizer) {
        return steps;
    }

    const auto& groups = optimizer->param_groups();
    const size_t count = std::min(groups.size(), kParameterGroupCount);
    for (size_t i = 0; i < count; ++i) {
        const auto& params = groups[i].params();
        if (params.empty() || !params[0].defined()) {
            continue;
        }
        const auto key = optimizer_state_key(optimizer.get(), params[0]);
        const auto state_it = optimizer->state().find(key);
        if (state_it != optimizer->state().end() && state_it->second) {
            steps[i] =
                static_cast<const torch::optim::AdamParamState&>(*state_it->second).step();
        }
    }
    return steps;
}

void GaussianModel::bind_active_capacity_views(const AdamSteps& steps)
{
    if (!_capacity_storage_enabled) {
        return;
    }
    if (_size < 0 || _size > _capacity) {
        throw std::runtime_error("Gaussian capacity invariant violated");
    }
    if (capacity_trace_enabled()) {
        capacity_trace(
            "rebind begin size=" + std::to_string(_size)
            + " capacity=" + std::to_string(_capacity));
    }

    if (optimizer) {
        optimizer->zero_grad(true);
        for (auto& group : optimizer->param_groups()) {
            if (!group.params().empty() && group.params()[0].defined()) {
                optimizer->state().erase(
                    optimizer_state_key(optimizer.get(), group.params()[0]));
            }
        }
    }

    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        if (!_parameter_storage[i].defined()) {
            parameter_view(i) = torch::Tensor();
            continue;
        }
        auto active =
            _parameter_storage[i].narrow(0, 0, _size).detach().set_requires_grad(true);
        parameter_view(i) = active;
        if (optimizer) {
            optimizer->param_groups()[i].params()[0] = active;
        }
    }

    if (!optimizer) {
        return;
    }

    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        auto& param = optimizer->param_groups()[i].params()[0];
        auto state = std::make_unique<torch::optim::AdamParamState>();
        state->step(steps[i]);
        state->exp_avg(
            _adam_exp_avg_storage[i].narrow(0, 0, _size).detach());
        state->exp_avg_sq(
            _adam_exp_avg_sq_storage[i].narrow(0, 0, _size).detach());
        optimizer->state()[optimizer_state_key(optimizer.get(), param)] = std::move(state);
    }
    if (capacity_trace_enabled()) {
        capacity_trace("rebind complete");
    }
}

void GaussianModel::initialize_capacity_storage(const TensorGroup& tensors)
{
    const int64_t count = tensors[0].size(0);
    uint64_t initial_payload = 0;
    _size = count;
    _capacity = 0;
    for (auto& tensor : _parameter_storage) {
        tensor = torch::Tensor();
    }
    for (auto& tensor : _adam_exp_avg_storage) {
        tensor = torch::Tensor();
    }
    for (auto& tensor : _adam_exp_avg_sq_storage) {
        tensor = torch::Tensor();
    }

    _capacity = choose_growth_capacity(count);
    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        auto sizes = tensors[i].sizes().vec();
        sizes[0] = _capacity;
        _parameter_storage[i] =
            torch::empty(sizes, tensors[i].options().requires_grad(false));
        _parameter_storage[i].narrow(0, 0, count).copy_(tensors[i].detach());
        initial_payload +=
            static_cast<uint64_t>(tensors[i].numel())
            * static_cast<uint64_t>(tensors[i].element_size());
        _adam_exp_avg_storage[i] =
            torch::zeros(sizes, tensors[i].options().requires_grad(false));
        _adam_exp_avg_sq_storage[i] =
            torch::zeros(sizes, tensors[i].options().requires_grad(false));
    }
    ++_reserve_count;
    bind_active_capacity_views(AdamSteps{});
    _last_payload_bytes_copied = initial_payload;
    _last_payload_bytes_avoided = 0;
    _payload_bytes_copied += initial_payload;
    mark_capacity_event("GS/capacity_init", initial_payload, 0);
}

bool GaussianModel::reserve_capacity(const int64_t required_size)
{
    if (required_size <= _capacity) {
        return false;
    }

    GOFUSION_NVTX_RANGE("GS/reserve_grow");
    const int64_t new_capacity = choose_growth_capacity(required_size);
    TensorGroup new_parameters;
    TensorGroup new_exp_avg;
    TensorGroup new_exp_avg_sq;

    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        auto sizes = _parameter_storage[i].sizes().vec();
        sizes[0] = new_capacity;
        const auto options = _parameter_storage[i].options().requires_grad(false);
        new_parameters[i] = torch::empty(sizes, options);
        new_exp_avg[i] = torch::zeros(sizes, options);
        new_exp_avg_sq[i] = torch::zeros(sizes, options);
        if (_size > 0) {
            new_parameters[i].narrow(0, 0, _size).copy_(
                _parameter_storage[i].narrow(0, 0, _size));
            if (_adam_exp_avg_storage[i].defined()) {
                new_exp_avg[i].narrow(0, 0, _size).copy_(
                    _adam_exp_avg_storage[i].narrow(0, 0, _size));
            }
            if (_adam_exp_avg_sq_storage[i].defined()) {
                new_exp_avg_sq[i].narrow(0, 0, _size).copy_(
                    _adam_exp_avg_sq_storage[i].narrow(0, 0, _size));
            }
        }
    }

    _parameter_storage = std::move(new_parameters);
    _adam_exp_avg_storage = std::move(new_exp_avg);
    _adam_exp_avg_sq_storage = std::move(new_exp_avg_sq);
    _capacity = new_capacity;
    ++_reserve_count;
    return true;
}

void GaussianModel::append_capacity_storage(const TensorGroup& tensors)
{
    const int64_t append_size = tensors[0].size(0);
    if (append_size <= 0) {
        return;
    }
    for (size_t i = 1; i < kParameterGroupCount; ++i) {
        if (tensors[i].size(0) != append_size) {
            throw std::runtime_error("Gaussian capacity append size mismatch");
        }
    }

    const int64_t old_size = _size;
    const int64_t required_size = old_size + append_size;
    const uint64_t full_payload = active_parameter_payload_bytes() * 3;
    uint64_t suffix_payload = 0;
    for (const auto& tensor : tensors) {
        suffix_payload +=
            static_cast<uint64_t>(tensor.numel())
            * static_cast<uint64_t>(tensor.element_size());
    }
    const AdamSteps steps = capture_adam_steps();
    if (optimizer) {
        optimizer->zero_grad(true);
    }

    const bool grew = reserve_capacity(required_size);
    {
        GOFUSION_NVTX_RANGE("GS/append_suffix_init");
        for (size_t i = 0; i < kParameterGroupCount; ++i) {
            _parameter_storage[i].narrow(0, old_size, append_size).copy_(
                tensors[i].detach());
            _adam_exp_avg_storage[i].narrow(0, old_size, append_size).zero_();
            _adam_exp_avg_sq_storage[i].narrow(0, old_size, append_size).zero_();
        }
    }

    _size = required_size;
    bind_active_capacity_views(steps);

    _last_payload_bytes_copied = suffix_payload + (grew ? full_payload : 0);
    _last_payload_bytes_avoided = grew ? 0 : full_payload;
    _payload_bytes_copied += _last_payload_bytes_copied;
    _payload_bytes_avoided += _last_payload_bytes_avoided;
    ++_append_count;
    mark_capacity_event(
        "GS/append", _last_payload_bytes_copied, _last_payload_bytes_avoided);
}

void GaussianModel::initialize_model_tensors(TensorGroup tensors)
{
    torch::NoGradGuard no_grad;
    optimizer.reset();

    if (_capacity_storage_enabled) {
        initialize_capacity_storage(tensors);
    }
    else {
        for (size_t i = 0; i < kParameterGroupCount; ++i) {
            parameter_view(i) =
                tensors[i].detach().contiguous().set_requires_grad(true);
        }
        _size = tensors[0].size(0);
        _capacity = _size;
    }

    training_setup();
    if (_capacity_storage_enabled) {
        bind_active_capacity_views(AdamSteps{});
    }
    _init_status = true;
}

void GaussianModel::append_model_tensors(TensorGroup tensors)
{
    GOFUSION_NVTX_RANGE("GS/append");
    torch::NoGradGuard no_grad;
    if (_capacity_storage_enabled) {
        append_capacity_storage(tensors);
        return;
    }

    if (optimizer) {
        optimizer->zero_grad(true);
    }
    cat_tensors_to_optimizer(optimizer.get(), tensors[0], _xyz, 0);
    cat_tensors_to_optimizer(optimizer.get(), tensors[1], _features_dc, 1);
    cat_tensors_to_optimizer(optimizer.get(), tensors[2], _features_rest, 2);
    cat_tensors_to_optimizer(optimizer.get(), tensors[3], _scaling, 3);
    cat_tensors_to_optimizer(optimizer.get(), tensors[4], _rotation, 4);
    cat_tensors_to_optimizer(optimizer.get(), tensors[5], _opacity, 5);
    _size = _xyz.size(0);
    _capacity = _size;
    ++_append_count;
}

void GaussianModel::training_setup()
{
    std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups;
    optimizer_params_groups.reserve(6);

    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_xyz}, std::make_unique<torch::optim::AdamOptions>(optimParams.position_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_features_dc}, std::make_unique<torch::optim::AdamOptions>(optimParams.feature_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_features_rest},
        std::make_unique<torch::optim::AdamOptions>(optimParams.feature_lr / 20.)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_scaling}, std::make_unique<torch::optim::AdamOptions>(optimParams.scaling_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_rotation}, std::make_unique<torch::optim::AdamOptions>(optimParams.rotation_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_opacity}, std::make_unique<torch::optim::AdamOptions>(optimParams.opacity_lr)));

    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[4].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[5].options()).eps(1e-15);

    optimizer = std::make_unique<torch::optim::Adam>(optimizer_params_groups,
                                                     torch::optim::AdamOptions(0.f).eps(1e-15));
}

namespace {
__global__ void fused_adam_step_kernel(const float* grad,
                                       float* param,
                                       float* exp_avg,
                                       float* exp_avg_sq,
                                       int n_elements,
                                       float step_size,
                                       float beta1,
                                       float beta2,
                                       float epsilon,
                                       float bias_correction2_sqrt_recip)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }

    const float g = grad[idx];
    const float m1_prev = exp_avg[idx];
    const float m2_prev = exp_avg_sq[idx];
    const float g_sq = g * g;
    const float m1 = fmaf(beta1, m1_prev - g, g);
    const float m2 = fmaf(beta2, m2_prev - g_sq, g_sq);
    const float denom = sqrtf(m2) * bias_correction2_sqrt_recip + epsilon;
    param[idx] -= step_size * (m1 / denom);
    exp_avg[idx] = m1;
    exp_avg_sq[idx] = m2;
}
} 

void fused_adam_step(torch::optim::Adam* optimizer)
{
    if (!optimizer) {
        return;
    }

    for (auto& group : optimizer->param_groups()) {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        if (options.weight_decay() != 0.0 || options.amsgrad()) {
            optimizer->step();
            return;
        }
    }

    for (auto& group : optimizer->param_groups()) {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        auto& params = group.params();
        if (params.empty()) {
            continue;
        }

        auto& param = params[0];
        auto grad = param.grad();
        if (!param.defined() || !grad.defined() || grad.numel() == 0) {
            continue;
        }
        if (!param.is_cuda() || !grad.is_cuda()) {
            continue;
        }
        if (param.scalar_type() != torch::kFloat32 || grad.scalar_type() != torch::kFloat32) {
            continue;
        }

        const int device_index = param.get_device();
        c10::cuda::CUDAGuard device_guard(device_index);
        const auto stream_object =
            c10::cuda::getCurrentCUDAStream(device_index);
        const cudaStream_t stream = stream_object.stream();

        const auto key = optimizer_state_key(optimizer, param);
        auto state_it = optimizer->state().find(key);
        if (state_it == optimizer->state().end()) {
            auto state = std::make_unique<torch::optim::AdamParamState>();
            state->step(0);
            state->exp_avg(torch::zeros_like(param));
            state->exp_avg_sq(torch::zeros_like(param));
            optimizer->state()[key] = std::move(state);
            state_it = optimizer->state().find(key);
        }

        auto& adam_state = static_cast<torch::optim::AdamParamState&>(*state_it->second);
        if (!adam_state.exp_avg().defined() || adam_state.exp_avg().numel() != param.numel()) {
            adam_state.exp_avg(torch::zeros_like(param));
        }
        if (!adam_state.exp_avg_sq().defined()
            || adam_state.exp_avg_sq().numel() != param.numel()) {
            adam_state.exp_avg_sq(torch::zeros_like(param));
        }

        const int64_t step_count = adam_state.step() + 1;
        adam_state.step(step_count);

        const auto betas = options.betas();
        const double beta1 = std::get<0>(betas);
        const double beta2 = std::get<1>(betas);
        const double bias_correction1_recip = 1.0 / (1.0 - std::pow(beta1, step_count));
        const double bias_correction2_sqrt_recip = 1.0 / std::sqrt(1.0 - std::pow(beta2, step_count));
        const float step_size = static_cast<float>(options.lr() * bias_correction1_recip);

        auto grad_contig = grad.contiguous();
        const int n_elements = static_cast<int>(param.numel());
        fused_adam_step_kernel<<<(n_elements + 255) / 256, 256, 0, stream>>>(
            grad_contig.data_ptr<float>(),
            param.data_ptr<float>(),
            adam_state.exp_avg().data_ptr<float>(),
            adam_state.exp_avg_sq().data_ptr<float>(),
            n_elements,
            step_size,
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(options.eps()),
            static_cast<float>(bias_correction2_sqrt_recip));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void GaussianModel::Save_ply_tensors_cpu(const PlyTensorsCPU& tensors,
                                         const std::filesystem::path& file_path,
                                         int frame)
{
    const auto require_cpu_float_matrix = [](const torch::Tensor& tensor,
                                             const char* name,
                                             const int64_t rows,
                                             const int64_t columns) {
        if (!tensor.defined() || tensor.device().is_cuda()
            || tensor.scalar_type() != torch::kFloat32 || !tensor.is_contiguous()
            || tensor.dim() != 2 || tensor.size(0) != rows || tensor.size(1) != columns) {
            std::ostringstream message;
            message << "Save_ply_tensors_cpu: " << name << " must be contiguous CPU float32 ("
                    << rows << "," << columns << ")";
            throw std::invalid_argument(message.str());
        }
    };

    if (!tensors.xyz.defined() || tensors.xyz.dim() != 2 || tensors.xyz.size(1) != 3
        || tensors.xyz.size(0) <= 0) {
        throw std::invalid_argument("Save_ply_tensors_cpu: xyz must be a non-empty (N,3) tensor");
    }
    const int64_t rows = tensors.xyz.size(0);
    require_cpu_float_matrix(tensors.xyz, "xyz", rows, 3);
    require_cpu_float_matrix(tensors.f_dc, "f_dc", rows, 3);
    if (!tensors.f_rest.defined() || tensors.f_rest.dim() != 2) {
        throw std::invalid_argument("Save_ply_tensors_cpu: f_rest must be a 2D tensor");
    }
    require_cpu_float_matrix(tensors.f_rest, "f_rest", rows, tensors.f_rest.size(1));
    require_cpu_float_matrix(tensors.opacity, "opacity", rows, 1);
    require_cpu_float_matrix(tensors.scale, "scale", rows, 3);
    require_cpu_float_matrix(tensors.rotation, "rotation", rows, 4);

    auto normals = torch::zeros_like(tensors.xyz);
    std::vector<torch::Tensor> tensor_attributes = {tensors.xyz,
                                                    normals,
                                                    tensors.f_dc,
                                                    tensors.f_rest,
                                                    tensors.opacity,
                                                    tensors.scale,
                                                    tensors.rotation};
    std::vector<std::string> attributes = {"x", "y", "z", "nx", "ny", "nz"};
    for (int64_t i = 0; i < tensors.f_dc.size(1); ++i) {
        attributes.push_back("f_dc_" + std::to_string(i));
    }
    for (int64_t i = 0; i < tensors.f_rest.size(1); ++i) {
        attributes.push_back("f_rest_" + std::to_string(i));
    }
    attributes.emplace_back("opacity");
    for (int64_t i = 0; i < tensors.scale.size(1); ++i) {
        attributes.push_back("scale_" + std::to_string(i));
    }
    for (int64_t i = 0; i < tensors.rotation.size(1); ++i) {
        attributes.push_back("rot_" + std::to_string(i));
    }

    const auto folder = file_path / ("iteration_" + std::to_string(frame));
    std::filesystem::create_directories(folder);
    Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
}

PlyTensorsCPU GaussianModel::Export_ply_tensors_cpu() const
{
    PlyTensorsCPU out;

    if (!_init_status || !_xyz.defined() || _xyz.numel() == 0) {
        out.xyz = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        out.f_dc = torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        out.f_rest = torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        out.opacity = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        out.scale = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        out.rotation = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        return out;
    }

    torch::NoGradGuard no_grad;

    out.xyz = _xyz.detach().to(torch::kCPU).contiguous();

    out.f_dc = _features_dc.detach().transpose(1, 2).flatten(1).to(torch::kCPU).contiguous();
    out.f_rest = _features_rest.detach().transpose(1, 2).flatten(1).to(torch::kCPU).contiguous();

    out.opacity = _opacity.detach().to(torch::kCPU).contiguous();
    if (out.opacity.dim() == 1) {
        out.opacity = out.opacity.view({-1, 1}).contiguous();
    }

    out.scale = _scaling.detach().to(torch::kCPU).contiguous();
    out.rotation = _rotation.detach().to(torch::kCPU).contiguous();

    return out;
}

void GaussianModel::reset_model_storage()
{
    if (optimizer) {
        optimizer->zero_grad(true);
    }
    optimizer.reset();
    _xyz = torch::Tensor();
    _features_dc = torch::Tensor();
    _features_rest = torch::Tensor();
    _scaling = torch::Tensor();
    _rotation = torch::Tensor();
    _opacity = torch::Tensor();
    for (auto& tensor : _parameter_storage) {
        tensor = torch::Tensor();
    }
    for (auto& tensor : _adam_exp_avg_storage) {
        tensor = torch::Tensor();
    }
    for (auto& tensor : _adam_exp_avg_sq_storage) {
        tensor = torch::Tensor();
    }
    _size = 0;
    _capacity = 0;
    _init_status = false;
}

void GaussianModel::shrink_capacity_to_size()
{
    if (!_capacity_storage_enabled || _size <= 0 || _capacity <= _size) {
        return;
    }

    GOFUSION_NVTX_RANGE("GS/capacity_shrink");
    TensorGroup compact_parameters;
    uint64_t copied_payload = 0;
    for (size_t i = 0; i < kParameterGroupCount; ++i) {
        auto sizes = _parameter_storage[i].sizes().vec();
        sizes[0] = _size;
        compact_parameters[i] =
            torch::empty(sizes, _parameter_storage[i].options().requires_grad(false));
        compact_parameters[i].copy_(_parameter_storage[i].narrow(0, 0, _size));
        copied_payload +=
            static_cast<uint64_t>(compact_parameters[i].numel())
            * static_cast<uint64_t>(compact_parameters[i].element_size());
    }
    _parameter_storage = std::move(compact_parameters);
    _capacity = _size;
    bind_active_capacity_views(AdamSteps{});
    _last_payload_bytes_copied = copied_payload;
    _last_payload_bytes_avoided = 0;
    _payload_bytes_copied += copied_payload;
    mark_capacity_event("GS/capacity_shrink", copied_payload, 0);
}

void GaussianModel::Release_training_state(const bool shrink_capacity)
{
    torch::NoGradGuard no_grad;
    if (optimizer) {
        optimizer->zero_grad(true);
    }
    optimizer.reset();
    for (auto& tensor : _adam_exp_avg_storage) {
        tensor = torch::Tensor();
    }
    for (auto& tensor : _adam_exp_avg_sq_storage) {
        tensor = torch::Tensor();
    }
    if (shrink_capacity) {
        shrink_capacity_to_size();
    }
}

void GaussianModel::Import_ply_tensors_cpu(const PlyTensorsCPU& t)
{
    torch::NoGradGuard no_grad;

    if (!t.xyz.defined() || t.xyz.numel() == 0) {
        reset_model_storage();
        return;
    }

    auto xyz_cpu = t.xyz.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto fdc_cpu = t.f_dc.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto frest_cpu = t.f_rest.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto opa_cpu = t.opacity.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto scl_cpu = t.scale.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto rot_cpu = t.rotation.to(torch::kCPU).to(torch::kFloat32).contiguous();

    if (opa_cpu.dim() == 1) {
        opa_cpu = opa_cpu.view({-1, 1}).contiguous();
    }

    const long N = xyz_cpu.size(0);
    const long SH = static_cast<long>(std::pow((_sh_degree + 1), 2));
    const long REST = std::max<long>(0, SH - 1);

    if (fdc_cpu.dim() != 2 || fdc_cpu.size(0) != N || fdc_cpu.size(1) != 3) {
        throw std::runtime_error("Import_ply_tensors_cpu: f_dc must be (N,3)");
    }
    if (REST > 0) {
        if (frest_cpu.dim() != 2 || frest_cpu.size(0) != N || frest_cpu.size(1) != 3 * REST) {
            throw std::runtime_error("Import_ply_tensors_cpu: f_rest must be (N, 3*(SH-1))");
        }
    }

    auto xyz = xyz_cpu.to(torch::kCUDA, true).contiguous();
    auto features_dc =
        fdc_cpu.view({N, 3, 1}).transpose(1, 2).to(torch::kCUDA, true).contiguous();
    torch::Tensor features_rest;
    if (REST > 0) {
        features_rest =
            frest_cpu.view({N, 3, REST})
                .transpose(1, 2)
                .to(torch::kCUDA, true)
                .contiguous();
    }
    else {
        features_rest =
            torch::empty({N, 0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                .contiguous();
    }
    auto opacity = opa_cpu.to(torch::kCUDA, true).contiguous();
    auto scaling = scl_cpu.to(torch::kCUDA, true).contiguous();
    auto rotation = rot_cpu.to(torch::kCUDA, true).contiguous();
    initialize_model_tensors(
        {xyz, features_dc, features_rest, scaling, rotation, opacity});
}

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              int param_position)
{
    if (!optimizer) {
        throw std::runtime_error("cat_tensors_to_optimizer: optimizer is not initialized");
    }
    const auto old_key =
        optimizer_state_key(optimizer, optimizer->param_groups()[param_position].params()[0]);
    auto state = std::make_unique<torch::optim::AdamParamState>();
    const auto state_it = optimizer->state().find(old_key);
    if (state_it != optimizer->state().end() && state_it->second) {
        const auto& old_state =
            static_cast<const torch::optim::AdamParamState&>(*state_it->second);
        state->step(old_state.step());
        state->exp_avg(torch::cat(
            {old_state.exp_avg(), torch::zeros_like(extension_tensor)}, 0));
        state->exp_avg_sq(torch::cat(
            {old_state.exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));
        optimizer->state().erase(state_it);
    }
    else {
        state->step(0);
        state->exp_avg(torch::zeros_like(
            torch::cat({old_tensor.detach(), extension_tensor.detach()}, 0)));
        state->exp_avg_sq(torch::zeros_like(state->exp_avg()));
    }

    optimizer->param_groups()[param_position].params()[0] =
        torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    const auto new_key =
        optimizer_state_key(optimizer, optimizer->param_groups()[param_position].params()[0]);
    optimizer->state()[new_key] = std::move(state);
}

} 
