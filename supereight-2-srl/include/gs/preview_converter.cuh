/*
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 */

#pragma once

#include <cstdint>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <torch/torch.h>

namespace gs {

namespace detail {
struct PreviewRgb8Slot;
}

struct PreviewRgb8CpuFrame {
    std::shared_ptr<detail::PreviewRgb8Slot> slot;
    int height = 0;
    int width = 0;

    bool valid() const
    {
        return slot && height > 0 && width > 0;
    }
};

struct PreviewConversionStatsSnapshot {
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t staging_grows = 0;
    uint64_t bytes_d2h = 0;
};

PreviewRgb8CpuFrame stagePreviewRgbHwcFloatToCpu(
    const torch::Tensor& rendered_rgb_hwc);

cv::Mat consumePreviewRgbCpu(PreviewRgb8CpuFrame frame);

PreviewConversionStatsSnapshot previewConversionStats();

} 
