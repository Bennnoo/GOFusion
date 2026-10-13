/*
 * SPDX-FileCopyrightText: 2023 Janusch Patas
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * All rights reserved. Some of the code is derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
 * gaussian() and create_window(): adapted from pytorch-ssim, Copyright Evan Su, 2017 (MIT).
 * See LICENSES/pytorch-ssim-MIT.txt at the repository root.
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#ifndef GS_GAUSSIAN_UTIL_HPP
#define GS_GAUSSIAN_UTIL_HPP

#include <Eigen/Dense>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <nvml.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <random>
#include <stdexcept>
#include <tinyply.h>
#include <torch/torch.h>

#include "gs/preview_converter.cuh"

using Slice = torch::indexing::Slice;

namespace gs {
struct Camera {
    int width = 0;
    int height = 0;
    float fov_x = 0.0f;
    float fov_y = 0.0f;
    float fx = -1.0f;
    float fy = -1.0f;
    float cx = -1.0f;
    float cy = -1.0f;
    torch::Tensor T_W2C;

    torch::Tensor T_W2C_row_major;
    torch::Tensor full_proj_matrix;

    torch::Tensor proj_matrix_raw;
    torch::Tensor cam_center;
};

inline void refreshFixedPoseCameraDerivedTensors(Camera& camera,
                                                 const torch::Tensor& T_W2C,
                                                 const torch::Tensor& projection)
{
    if (!T_W2C.defined() || T_W2C.dim() != 2 || T_W2C.size(0) != 4
        || T_W2C.size(1) != 4) {
        throw std::invalid_argument("Fixed-pose camera requires a 4x4 T_W2C tensor");
    }
    if (!projection.defined() || projection.dim() != 2 || projection.size(0) != 4
        || projection.size(1) != 4) {
        throw std::invalid_argument("Fixed-pose camera requires a 4x4 projection tensor");
    }

    torch::NoGradGuard no_grad;
    camera.T_W2C = T_W2C.contiguous();
    camera.proj_matrix_raw =
        projection
            .to(camera.T_W2C.device(), camera.T_W2C.scalar_type(), true, false)
            .contiguous();
    camera.T_W2C_row_major = camera.T_W2C.transpose(0, 1).contiguous();
    camera.full_proj_matrix =
        camera.T_W2C.mm(camera.proj_matrix_raw).contiguous();
    camera.cam_center =
        camera.T_W2C.inverse()[3].slice(0, 0, 3).contiguous();
}

inline void refreshFixedPoseCameraDerivedTensors(Camera& camera,
                                                 const torch::Tensor& projection)
{
    refreshFixedPoseCameraDerivedTensors(camera, camera.T_W2C, projection);
}

inline const torch::Tensor& fixedPoseW2CRowMajor(Camera& camera)
{
    const bool cache_invalid =
        !camera.T_W2C_row_major.defined()
        || camera.T_W2C_row_major.dim() != 2
        || camera.T_W2C_row_major.size(0) != 4
        || camera.T_W2C_row_major.size(1) != 4
        || camera.T_W2C_row_major.device() != camera.T_W2C.device()
        || camera.T_W2C_row_major.scalar_type() != camera.T_W2C.scalar_type();
    if (cache_invalid) {
        torch::NoGradGuard no_grad;
        camera.T_W2C_row_major = camera.T_W2C.transpose(0, 1).contiguous();
    }
    return camera.T_W2C_row_major;
}

struct DataPacket {
    int ID = -1;
    int global_iter = -1;
    int num_kf = 0;
    int num_splats = 0;
    float fps = 0.0f;


    cv::Mat rgb;
    cv::Mat depth;

    cv::Mat rendered_rgb;

    PreviewRgb8CpuFrame rendered_rgb_staging;
};

class DataQueue {
    public:
    struct Stats {
        uint64_t submitted = 0;
        uint64_t replaced = 0;
        uint64_t popped = 0;
        uint64_t high_water = 0;
    };

    bool push(DataPacket data)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        if (shutdown_) {
            return false;
        }
        ++submitted_;
        while (!_queue.empty()) {
            _queue.pop();
            ++replaced_;
        }
        _queue.push(std::move(data));
        const uint64_t queued = static_cast<uint64_t>(_queue.size());
        if (queued > high_water_) {
            high_water_ = queued;
        }
        _cv.notify_one();
        return true;
    }

    bool pop(DataPacket* data)
    {
        std::unique_lock<std::mutex> lock(_mtx);
        _cv.wait(lock, [this]() { return shutdown_ || !_queue.empty(); });
        if (_queue.empty()) {
            return false;
        }
        *data = std::move(_queue.front());
        _queue.pop();
        ++popped_;
        return true;
    }

    int getSize()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return _queue.size();
    }

    Stats stats()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return Stats{submitted_, replaced_, popped_, high_water_};
    }

    void shutdown()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        shutdown_ = true;
        _cv.notify_all();
    }

    void resume()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        shutdown_ = false;
    }

    private:
    std::queue<DataPacket> _queue;
    std::mutex _mtx;
    std::condition_variable _cv;
    bool shutdown_ = false;
    uint64_t submitted_ = 0;
    uint64_t replaced_ = 0;
    uint64_t popped_ = 0;
    uint64_t high_water_ = 0;
};

static inline torch::Tensor gaussian(int window_size, float sigma)
{
    torch::Tensor gauss = torch::empty(window_size);
    for (int x = 0; x < window_size; ++x) {
        gauss[x] = std::exp(-(std::pow(static_cast<float>(x - std::floor(window_size / 2.f)), 2))
                            / (2.f * sigma * sigma));
    }
    return gauss / gauss.sum();
}

static inline torch::Tensor create_window(int window_size, int channel)
{
    auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
    return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
}

const int window_size = 11;
const int channel = 3;
const auto conv_window =
    create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);

static const double C0 = 0.28209479177387814;
static const double C1 = 0.4886025119029199;
static std::vector<double> C2 = {1.0925484305920792,
                                 -1.0925484305920792,
                                 0.31539156525252005,
                                 -1.0925484305920792,
                                 0.5462742152960396};
static std::vector<double> C3 = {-0.5900435899266435,
                                 2.890611442640554,
                                 -0.4570457994644658,
                                 0.3731763325901154,
                                 -0.4570457994644658,
                                 1.445305721320277,
                                 -0.5900435899266435};
static std::vector<double> C4 = {2.5033429417967046,
                                 -1.7701307697799304,
                                 0.9461746957575601,
                                 -0.6690465435572892,
                                 0.10578554691520431,
                                 -0.6690465435572892,
                                 0.47308734787878004,
                                 -1.7701307697799304,
                                 0.6258357354491761};

static inline torch::Tensor RGB2SH(const torch::Tensor& rgb)
{
    return (rgb - 0.5f) / static_cast<float>(C0);
}

struct Point {
    float x;
    float y;
    float z;
};

struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct PoseUpdate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t sample_index = std::numeric_limits<size_t>::max();
    bool is_current_frame = false;
    Eigen::Matrix4f T_WC = Eigen::Matrix4f::Identity();
};

using PoseUpdateVector = std::vector<PoseUpdate, Eigen::aligned_allocator<PoseUpdate>>;

struct FisherTrainingDiag {
    torch::Tensor xyz;
    torch::Tensor opacity;
};

struct FisherTrainingSample {
    bool is_current_frame = false;
    bool contributes_to_history = false;
    FisherTrainingDiag diag;
};

using FisherTrainingSampleVector = std::vector<FisherTrainingSample>;

static inline torch::Tensor inverse_sigmoid(torch::Tensor x)
{
    return torch::log(x / (1 - x));
}

static inline torch::Tensor strip_lowerdiag(torch::Tensor L)
{
    torch::Tensor uncertainty =
        torch::zeros({L.size(0), 6}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    uncertainty.index_put_({Slice(), 0}, L.index({Slice(), 0, 0}));
    uncertainty.index_put_({Slice(), 1}, L.index({Slice(), 0, 1}));
    uncertainty.index_put_({Slice(), 2}, L.index({Slice(), 0, 2}));
    uncertainty.index_put_({Slice(), 3}, L.index({Slice(), 1, 1}));
    uncertainty.index_put_({Slice(), 4}, L.index({Slice(), 1, 2}));
    uncertainty.index_put_({Slice(), 5}, L.index({Slice(), 2, 2}));
    return uncertainty;
}

static inline torch::Tensor strip_symmetric(torch::Tensor sym)
{
    return strip_lowerdiag(sym);
}

static inline torch::Tensor
getProjectionMatrix(int W, int H, float fx, float fy, float cx, float cy)
{
    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 2 * fx / W;
    P(1, 1) = 2 * fy / H;
    P(0, 2) = 2 * (cx / W) - 1;
    P(1, 2) = 2 * (cy / H) - 1;
    P(2, 2) = 0;
    P(2, 3) = 0;
    P(3, 2) = 1;

    auto PTensor = torch::from_blob(P.data(), {4, 4}, torch::kFloat);
    return PTensor.clone();
}

static inline void Write_output_ply(const std::filesystem::path& file_path,
                                    const std::vector<torch::Tensor>& tensors,
                                    const std::vector<std::string>& attribute_names)
{
    tinyply::PlyFile plyFile;

    size_t attribute_offset = 0; 

    for (size_t i = 0; i < tensors.size(); ++i) {

        size_t columns = tensors[i].size(1);

        std::vector<std::string> current_attributes;
        for (size_t j = 0; j < columns; ++j) {
            current_attributes.push_back(attribute_names[attribute_offset + j]);
        }

        plyFile.add_properties_to_element("vertex",
                                          current_attributes,
                                          tinyply::Type::FLOAT32,
                                          tensors[i].size(0),
                                          reinterpret_cast<uint8_t*>(tensors[i].data_ptr<float>()),
                                          tinyply::Type::INVALID,
                                          0);

        attribute_offset += columns; 
    }

    std::filebuf fb;
    fb.open(file_path, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);
    plyFile.write(outputStream, true); 
}

static inline float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img)
{
    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

static inline std::vector<int> get_random_indices(int max_index)
{
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);

    static thread_local std::mt19937 rng([]() {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        return std::mt19937(seq);
    }());

    std::shuffle(indices.begin(), indices.end(), rng);
    std::reverse(indices.begin(), indices.end());
    return indices;
}

static inline auto getGPUMemoryUsage(int gpu_id = 0)
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlMemory_t memory;

    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << "\n";
        return ULLONG_MAX;
    }

    result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get handle for device " << std::to_string(gpu_id) << ": "
                  << nvmlErrorString(result) << "\n";
        nvmlShutdown();
        return ULLONG_MAX;
    }

    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get memory info: " << nvmlErrorString(result) << "\n";
        nvmlShutdown();
        return ULLONG_MAX;
    }

    auto used_gpu_memory = memory.used / (1024 * 1024);

    nvmlShutdown();

    return used_gpu_memory;
}
} 

#endif 
