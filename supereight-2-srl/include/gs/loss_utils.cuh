/*
 * Copyright (c) 2023 Janusch Patas.
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
 * ssim(): adapted from pytorch-ssim, Copyright Evan Su, 2017 (MIT).
 * See LICENSES/pytorch-ssim-MIT.txt at the repository root.
 */

#pragma once
#include <cmath>
#include <torch/torch.h>

#include "gaussian_utils.cuh"

namespace F = torch::nn::functional;

namespace gs {
static const float V1 = 0.01 * 0.01;
static const float V2 = 0.03 * 0.03;

static inline torch::Tensor l1_loss(const torch::Tensor& network_output, const torch::Tensor& gt)
{
    return torch::abs((network_output - gt)).mean();
}


static inline torch::Tensor ssim(const torch::Tensor& img1,
                                 const torch::Tensor& img2,
                                 const torch::Tensor& window,
                                 int window_size,
                                 int channel)
{
    auto mu1 =
        F::conv2d(img1, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu1_sq = mu1.pow(2);
    auto sigma1_sq = F::conv2d(img1 * img1,
                               window,
                               F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
        - mu1_sq;

    auto mu2 =
        F::conv2d(img2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu2_sq = mu2.pow(2);
    auto sigma2_sq = F::conv2d(img2 * img2,
                               window,
                               F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
        - mu2_sq;

    auto mu1_mu2 = mu1 * mu2;
    auto sigma12 = F::conv2d(img1 * img2,
                             window,
                             F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
        - mu1_mu2;
    auto ssim_map = ((2.f * mu1_mu2 + V1) * (2.f * sigma12 + V2))
        / ((mu1_sq + mu2_sq + V1) * (sigma1_sq + sigma2_sq + V2));

    return ssim_map.mean();
}

} 
