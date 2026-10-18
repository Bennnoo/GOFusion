/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>

namespace gs::surfel {

constexpr float kMinQuaternionNormSq = 1.0e-8f;
constexpr float kMinRayPlaneDenominator = 1.0e-6f;

__device__ inline float3 quaternionNormal(const float4 q, bool* valid = nullptr)
{
    const float norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    const bool is_valid = isfinite(norm_sq) && norm_sq > kMinQuaternionNormSq;
    if (valid) {
        *valid = is_valid;
    }
    if (!is_valid) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    const float inv = 1.0f / norm_sq;
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    return make_float3(2.0f * (x * z + r * y) * inv,
                       2.0f * (y * z - r * x) * inv,
                       (r * r - x * x - y * y + z * z) * inv);
}

__device__ inline float4 quaternionNormalBackward(const float4 q, const float3 grad_normal)
{
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    const float norm_sq = r * r + x * x + y * y + z * z;
    if (!(norm_sq > kMinQuaternionNormSq) || !isfinite(norm_sq)) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float3 numerator = make_float3(2.0f * (x * z + r * y),
                                         2.0f * (y * z - r * x),
                                         r * r - x * x - y * y + z * z);
    const float inv = 1.0f / norm_sq;
    const float inv_sq = inv * inv;
    const float grad_norm_sq = -(grad_normal.x * numerator.x
                                 + grad_normal.y * numerator.y
                                 + grad_normal.z * numerator.z)
        * inv_sq;
    const float3 grad_numerator = make_float3(grad_normal.x * inv,
                                               grad_normal.y * inv,
                                               grad_normal.z * inv);

    return make_float4(
        2.0f * y * grad_numerator.x - 2.0f * x * grad_numerator.y
            + 2.0f * r * grad_numerator.z + 2.0f * r * grad_norm_sq,
        2.0f * z * grad_numerator.x - 2.0f * r * grad_numerator.y
            - 2.0f * x * grad_numerator.z + 2.0f * x * grad_norm_sq,
        2.0f * r * grad_numerator.x + 2.0f * z * grad_numerator.y
            - 2.0f * y * grad_numerator.z + 2.0f * y * grad_norm_sq,
        2.0f * x * grad_numerator.x + 2.0f * y * grad_numerator.y
            + 2.0f * z * grad_numerator.z + 2.0f * z * grad_norm_sq);
}

__device__ inline float rayPlaneDepth(const float4 plane, const float3 ray)
{
    const float denominator = plane.x * ray.x + plane.y * ray.y + plane.z * ray.z;
    if (!isfinite(denominator) || fabsf(denominator) <= kMinRayPlaneDenominator) {
        return 0.0f;
    }
    const float depth = plane.w / denominator;
    return isfinite(depth) && depth > 0.0f ? depth : 0.0f;
}

__device__ inline float4 rayPlaneDepthBackward(const float4 plane,
                                               const float3 ray,
                                               const float grad_depth)
{
    const float denominator = plane.x * ray.x + plane.y * ray.y + plane.z * ray.z;
    if (!isfinite(denominator) || fabsf(denominator) <= kMinRayPlaneDenominator) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float depth = plane.w / denominator;
    if (!isfinite(depth) || depth <= 0.0f) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float grad_h = grad_depth / denominator;
    const float grad_denominator = -grad_depth * plane.w / (denominator * denominator);
    return make_float4(grad_denominator * ray.x,
                       grad_denominator * ray.y,
                       grad_denominator * ray.z,
                       grad_h);
}

} 
