/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace gs {

struct ErrorDensifyParams {
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

} // namespace gs
