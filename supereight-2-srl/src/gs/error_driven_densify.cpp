/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <torch/torch.h>

#include "gs/error_driven_densify.cuh"

namespace gs {

static inline double clamp01(double p)
{
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

DensifyMask build_densify_mask(const torch::Tensor& image_detached,
                                        const torch::Tensor& gt_img,
                                        const torch::Tensor& valid_mask,
                                        const ErrorDensifyParams& cfg)
{
    DensifyMask out;

    if (!cfg.enabled || cfg.tile_size <= 0) {
        return out;
    }
    if (!image_detached.defined() || !gt_img.defined()) {
        return out;
    }
    if (image_detached.numel() == 0 || gt_img.numel() == 0) {
        return out;
    }

    auto img = image_detached.contiguous();
    auto gt = gt_img.contiguous();

    auto err2 = (img - gt).pow(2).mean(0); // (H,W)
    const int64_t H = err2.size(0);
    const int64_t W = err2.size(1);
    if (H <= 0 || W <= 0) {
        return out;
    }

    auto bopts = torch::TensorOptions().dtype(torch::kBool).device(err2.device());
    torch::Tensor valid_bool;
    if (valid_mask.defined() && valid_mask.numel() > 0) {
        valid_bool = valid_mask.to(err2.device(), true).contiguous();
        if (valid_bool.scalar_type() != torch::kBool) {
            valid_bool = valid_bool > 0.5f;
        }
        if (valid_bool.dim() == 3 && valid_bool.size(0) == 1) {
            valid_bool = valid_bool.squeeze(0);
        }
        if (valid_bool.dim() != 2 || valid_bool.size(0) != H || valid_bool.size(1) != W) {
            return out;
        }
    }
    else {
        valid_bool = torch::ones({H, W}, bopts);
    }

    const int64_t ts = static_cast<int64_t>(cfg.tile_size);
    const int64_t pad_h = (ts - (H % ts)) % ts;
    const int64_t pad_w = (ts - (W % ts)) % ts;

    auto err2_select = err2.masked_fill(valid_bool.logical_not(), -1.0f);
    auto err4 = err2_select.unsqueeze(0).unsqueeze(0); // (1,1,H,W)
    auto valid4 = valid_bool.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    if (pad_h != 0 || pad_w != 0) {
        err4 = torch::nn::functional::pad(
            err4,
            torch::nn::functional::PadFuncOptions({0, pad_w, 0, pad_h})
                .mode(torch::kConstant)
                .value(-1.0));
        valid4 = torch::nn::functional::pad(
            valid4,
            torch::nn::functional::PadFuncOptions({0, pad_w, 0, pad_h})
                .mode(torch::kConstant)
                .value(0.0));
    }
    auto err2_pad = err4.squeeze(0).squeeze(0).contiguous(); // (Hpad,Wpad)
    auto valid_pad = valid4.squeeze(0).squeeze(0).contiguous(); // (Hpad,Wpad)

    const int64_t Hpad = H + pad_h;
    const int64_t Wpad = W + pad_w;
    const int64_t Ht = Hpad / ts;
    const int64_t Wt = Wpad / ts;

    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(err2.device());

    auto tiles = err2_pad.view({Ht, ts, Wt, ts}).permute({0, 2, 1, 3}).contiguous();
    auto tiles_flat = tiles.view({Ht * Wt, ts * ts});
    auto valid_tiles = valid_pad.view({Ht, ts, Wt, ts})
                           .permute({0, 2, 1, 3})
                           .contiguous()
                           .view({Ht * Wt, ts * ts})
                           .sum(1)
                           > 0.0f;

    auto mm = tiles_flat.max(1);
    auto tile_scores = std::get<0>(mm); // (Ht*Wt)
    auto tile_argmax = std::get<1>(mm); // (Ht*Wt)
    tile_scores = tile_scores.masked_fill(valid_tiles.logical_not(), -1.0f);

    const int64_t num_valid_tiles = valid_tiles.to(torch::kLong).sum().item<int64_t>();
    const double tile_ratio = clamp01(static_cast<double>(cfg.top_tiles_ratio));
    if (tile_ratio <= 0.0 || num_valid_tiles <= 0) {
        out.mask = torch::zeros({H, W}, fopts);
        return out;
    }

    int64_t K = static_cast<int64_t>(std::ceil(tile_ratio * static_cast<double>(num_valid_tiles)));
    if (K < 1) K = 1;
    if (K > num_valid_tiles) K = num_valid_tiles;

    auto tk = torch::topk(tile_scores, K);
    auto top_tile_idx = std::get<1>(tk); // (K)

    auto sel_argmax = tile_argmax.index_select(0, top_tile_idx); // (K)

    auto ty = top_tile_idx.floor_divide(Wt);
    auto tx = top_tile_idx.remainder(Wt);
    auto dy = sel_argmax.floor_divide(ts);
    auto dx = sel_argmax.remainder(ts);

    auto peak_x = (tx * ts + dx).clamp(0, W - 1).to(torch::kLong);
    auto peak_y = (ty * ts + dy).clamp(0, H - 1).to(torch::kLong);

    out.peak_x = peak_x;
    out.peak_y = peak_y;

    const int64_t patch_half = std::min<int64_t>(2, std::max<int64_t>(0, ts / 2));

    std::vector<int64_t> du, dv;
    du.reserve(static_cast<size_t>((2 * patch_half + 1) * (2 * patch_half + 1)));
    dv.reserve(static_cast<size_t>((2 * patch_half + 1) * (2 * patch_half + 1)));
    for (int64_t yy = -patch_half; yy <= patch_half; ++yy) {
        for (int64_t xx = -patch_half; xx <= patch_half; ++xx) {
            dv.push_back(yy);
            du.push_back(xx);
        }
    }
    auto du_t = torch::from_blob(du.data(), {static_cast<long>(du.size())},
                                 torch::TensorOptions().dtype(torch::kLong)).clone().to(err2.device());
    auto dv_t = torch::from_blob(dv.data(), {static_cast<long>(dv.size())},
                                 torch::TensorOptions().dtype(torch::kLong)).clone().to(err2.device());

    auto u_all = (peak_x.unsqueeze(1) + du_t.unsqueeze(0)).clamp(0, W - 1);
    auto v_all = (peak_y.unsqueeze(1) + dv_t.unsqueeze(0)).clamp(0, H - 1);

    auto lin = (v_all * W + u_all).reshape({-1}).to(torch::kLong);
    auto mask_flat = torch::zeros({H * W}, fopts);
    mask_flat.index_put_({lin}, 1.0f);
    out.mask = mask_flat.view({H, W}).mul(valid_bool.to(torch::kFloat32));

    return out;
}

torch::Tensor densify_region_mse(const torch::Tensor& image,
                                 const torch::Tensor& gt_img,
                                 const torch::Tensor& mask)
{
    if (!image.defined() || !gt_img.defined() || !mask.defined()) {
        return torch::Tensor();
    }
    if (image.numel() == 0 || gt_img.numel() == 0 || mask.numel() == 0) {
        return torch::Tensor();
    }

    auto err2 = (image - gt_img).pow(2).mean(0); // (H,W)
    auto denom = mask.sum();
    return (err2 * mask).sum() / (denom + 1.0e-6f);
}

} // namespace gs
