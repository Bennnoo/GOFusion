/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#ifndef MONO_CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define MONO_CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// Numeric stability thresholds (aligned with FasterGS-style behavior).
#define MONO_ALPHA_THRESHOLD (1.0f / 255.0f)
#define MONO_TRANSMITTANCE_THRESHOLD (1.0e-4f)
#define MONO_MIN_COV2D_DETERMINANT (1.0e-12f)
#define MONO_ONE_MINUS_ALPHA_EPS (1.0e-6f)
#define MONO_MIN_QUAT_NORM2 (1.0e-12f)

#endif
