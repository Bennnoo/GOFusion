/*
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 */

#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "gs/rgb_input_view.hpp"

namespace gs {

enum class RgbIngestMode {
    Auto,
    Mapped,
    Staged,
    HostCopy,
};

enum class RgbIngestBackend {
    Mapped,
    Staged,
    HostCopy,
};

struct RgbIngestResult {
    torch::Tensor image;
    RgbIngestBackend backend = RgbIngestBackend::HostCopy;

    torch::Tensor host_cpu_image;
};

RgbIngestMode configuredRgbIngestMode();

RgbIngestResult ingestRgb(const RgbInputView& input);
RgbIngestResult ingestRgb(const RgbInputView& input, RgbIngestMode mode);

torch::Tensor makeRgbCpuSnapshot(const RgbInputView& input);
torch::Tensor makeRgbCpuSnapshot(const torch::Tensor& cuda_rgb_chw);

} 
