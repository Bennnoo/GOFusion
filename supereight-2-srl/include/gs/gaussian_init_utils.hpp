/*
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>

#include "gs/gaussian.cuh"

namespace gs {

struct OrientedGaussianInit {
    gs::Point pos_w;     
    gs::Point scale_xyz; 
    gs::Quat rot_wxyz;   
};

static inline Eigen::Vector3f backproject_xyz_C(float u,
                                                float v,
                                                float d,
                                                float fx,
                                                float fy,
                                                float cx,
                                                float cy)
{
    const float x = (u - cx) / fx * d;
    const float y = (v - cy) / fy * d;
    return Eigen::Vector3f(x, y, d);
}

static inline float depth_at(const float* depth, int W, int H, int u, int v)
{
    u = std::clamp(u, 0, W - 1);
    v = std::clamp(v, 0, H - 1);
    return depth[u + W * v];
}

static inline Eigen::Vector3f depth_normal_C_precomp_inv_d0(const float* depth,
                                                            int W,
                                                            int H,
                                                            int u,
                                                            int v,
                                                            float d0,
                                                            float near_plane,
                                                            float inv_fx,
                                                            float inv_fy,
                                                            float cx,
                                                            float cy)
{
    if (!(d0 > near_plane)) {
        return Eigen::Vector3f(0.f, 0.f, 1.f);
    }

    const float dL = depth_at(depth, W, H, u - 1, v);
    const float dR = depth_at(depth, W, H, u + 1, v);
    const float dU = depth_at(depth, W, H, u, v - 1);
    const float dD = depth_at(depth, W, H, u, v + 1);

    const float dl = (dL > near_plane) ? dL : d0;
    const float dr = (dR > near_plane) ? dR : d0;
    const float du = (dU > near_plane) ? dU : d0;
    const float dd = (dD > near_plane) ? dD : d0;

    const float uC = float(u) + 0.5f;
    const float vC = float(v) + 0.5f;
    const float uL = float(u - 1) + 0.5f;
    const float uR = float(u + 1) + 0.5f;
    const float vU = float(v - 1) + 0.5f;
    const float vD = float(v + 1) + 0.5f;

    const float dx_x = inv_fx * ((uR - cx) * dr - (uL - cx) * dl);
    const float dx_y = inv_fy * (vC - cy) * (dr - dl);
    const float dx_z = dr - dl;

    const float dy_x = inv_fx * (uC - cx) * (dd - du);
    const float dy_y = inv_fy * ((vD - cy) * dd - (vU - cy) * du);
    const float dy_z = dd - du;

    const float nx = dx_y * dy_z - dx_z * dy_y;
    const float ny = dx_z * dy_x - dx_x * dy_z;
    const float nz = dx_x * dy_y - dx_y * dy_x;

    const float nn2 = nx * nx + ny * ny + nz * nz;
    if (!(nn2 > 1.0e-24f) || !std::isfinite(nn2)) {
        return Eigen::Vector3f(0.f, 0.f, 1.f);
    }

    const float invn = 1.0f / std::sqrt(nn2);
    return Eigen::Vector3f(nx * invn, ny * invn, nz * invn);
}

static inline Eigen::Vector3f depth_normal_C_precomp_inv(const float* depth,
                                                         int W,
                                                         int H,
                                                         int u,
                                                         int v,
                                                         float near_plane,
                                                         float inv_fx,
                                                         float inv_fy,
                                                         float cx,
                                                         float cy)
{
    return depth_normal_C_precomp_inv_d0(depth,
                                         W,
                                         H,
                                         u,
                                         v,
                                         depth_at(depth, W, H, u, v),
                                         near_plane,
                                         inv_fx,
                                         inv_fy,
                                         cx,
                                         cy);
}

static inline Eigen::Vector3f depth_normal_C(const float* depth,
                                             int W,
                                             int H,
                                             int u,
                                             int v,
                                             float near_plane,
                                             float fx,
                                             float fy,
                                             float cx,
                                             float cy)
{
    const float inv_fx = 1.0f / fx;
    const float inv_fy = 1.0f / fy;
    return depth_normal_C_precomp_inv(depth, W, H, u, v, near_plane, inv_fx, inv_fy, cx, cy);
}

static inline gs::Quat quat_wxyz_from_rotm_fast(float m00, float m01, float m02,
                                                float m10, float m11, float m12,
                                                float m20, float m21, float m22)
{
    gs::Quat q;
    const float tr = m00 + m11 + m22;

    if (tr > 0.f) {
        const float s = std::sqrt(std::max(0.f, tr + 1.f)) * 2.f;
        q.w = 0.25f * s;
        const float inv = (s > 1.0e-12f) ? (1.0f / s) : 0.f;
        q.x = (m21 - m12) * inv;
        q.y = (m02 - m20) * inv;
        q.z = (m10 - m01) * inv;
    }
    else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(std::max(0.f, 1.f + m00 - m11 - m22)) * 2.f;
        q.x = 0.25f * s;
        const float inv = (s > 1.0e-12f) ? (1.0f / s) : 0.f;
        q.w = (m21 - m12) * inv;
        q.y = (m01 + m10) * inv;
        q.z = (m02 + m20) * inv;
    }
    else if (m11 > m22) {
        const float s = std::sqrt(std::max(0.f, 1.f + m11 - m00 - m22)) * 2.f;
        q.y = 0.25f * s;
        const float inv = (s > 1.0e-12f) ? (1.0f / s) : 0.f;
        q.w = (m02 - m20) * inv;
        q.x = (m01 + m10) * inv;
        q.z = (m12 + m21) * inv;
    }
    else {
        const float s = std::sqrt(std::max(0.f, 1.f + m22 - m00 - m11)) * 2.f;
        q.z = 0.25f * s;
        const float inv = (s > 1.0e-12f) ? (1.0f / s) : 0.f;
        q.w = (m10 - m01) * inv;
        q.x = (m02 + m20) * inv;
        q.y = (m12 + m21) * inv;
    }

    const float n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (n2 > 1.0e-24f && std::isfinite(n2)) {
        const float invn = 1.0f / std::sqrt(n2);
        q.w *= invn;
        q.x *= invn;
        q.y *= invn;
        q.z *= invn;
    }
    else {
        q.w = 1.f;
        q.x = 0.f;
        q.y = 0.f;
        q.z = 0.f;
    }
    return q;
}

static inline gs::Quat quat_wxyz_from_basis_fast(const Eigen::Vector3f& x_w,
                                                 const Eigen::Vector3f& y_w,
                                                 const Eigen::Vector3f& z_w)
{

    const float m00 = x_w.x(), m01 = y_w.x(), m02 = z_w.x();
    const float m10 = x_w.y(), m11 = y_w.y(), m12 = z_w.y();
    const float m20 = x_w.z(), m21 = y_w.z(), m22 = z_w.z();
    return quat_wxyz_from_rotm_fast(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

static inline Eigen::Quaternionf quat_from_basis(const Eigen::Vector3f& x_w,
                                                 const Eigen::Vector3f& y_w,
                                                 const Eigen::Vector3f& z_w)
{
    const gs::Quat q = quat_wxyz_from_basis_fast(x_w, y_w, z_w);
    Eigen::Quaternionf eq(q.w, q.x, q.y, q.z);
    eq.normalize();
    return eq;
}

static inline bool make_gaussian_center_from_valid_depth_pixel_precomp(
    gs::Point* pos_w,
    int u,
    int v,
    float d0,
    float inv_fx,
    float inv_fy,
    float cx,
    float cy,
    const Eigen::Matrix3f& R_WC,
    const Eigen::Vector3f& t_WC)
{
    if (!pos_w || !(d0 > 0.f) || !std::isfinite(d0)) {
        return false;
    }

    const float uf = float(u) + 0.5f;
    const float vf = float(v) + 0.5f;
    const Eigen::Vector3f pC((uf - cx) * inv_fx * d0,
                             (vf - cy) * inv_fy * d0,
                             d0);
    const Eigen::Vector3f pW = R_WC * pC + t_WC;

    if (!pW.allFinite()) {
        return false;
    }
    pos_w->x = pW.x();
    pos_w->y = pW.y();
    pos_w->z = pW.z();
    return true;
}

static inline bool finish_oriented_gaussian_from_depth_pixel_precomp(
    OrientedGaussianInit* out,
    const gs::Point& pos_w,
    int u,
    int v,
    float leaf_w_px,
    float leaf_h_px,
    float d0,
    const float* depth,
    int W,
    int H,
    float near_plane,
    float inv_fx,
    float inv_fy,
    float cx,
    float cy,
    const Eigen::Matrix3f& R_WC,
    const Eigen::Vector3f& exW,
    const Eigen::Vector3f& eyW,
    float scale_mul_xy,
    float z_scale)
{
    if (!out || !depth || W <= 0 || H <= 0 || !(d0 > near_plane)
        || !std::isfinite(d0)) {
        return false;
    }

    u = std::clamp(u, 0, W - 1);
    v = std::clamp(v, 0, H - 1);
    out->pos_w = pos_w;

    Eigen::Vector3f nC = depth_normal_C_precomp_inv_d0(
        depth, W, H, u, v, d0, near_plane, inv_fx, inv_fy, cx, cy);

    Eigen::Vector3f nW = R_WC * nC;
    const float nwn = nW.norm();
    if (nwn > 1.0e-12f && !std::isnan(nwn)) {
        nW /= nwn;
    }
    else {
        nW = Eigen::Vector3f(0.f, 0.f, 1.f);
    }

    Eigen::Vector3f t1 = exW - exW.dot(nW) * nW;
    float t1n = t1.norm();
    if (!(t1n > 1.0e-12f) || std::isnan(t1n)) {
        t1 = eyW - eyW.dot(nW) * nW;
        t1n = t1.norm();
    }
    if (t1n > 1.0e-12f && !std::isnan(t1n)) {
        t1 /= t1n;
    }
    else {
        t1 = Eigen::Vector3f(1.f, 0.f, 0.f);
    }

    Eigen::Vector3f t2 = nW.cross(t1);
    const float t2n = t2.norm();
    if (t2n > 1.0e-12f && !std::isnan(t2n)) {
        t2 /= t2n;
    }
    else {
        t2 = Eigen::Vector3f(0.f, 1.f, 0.f);
    }

    out->rot_wxyz = quat_wxyz_from_basis_fast(t1, t2, nW);

    const float lw = (leaf_w_px > 0.f && std::isfinite(leaf_w_px)) ? leaf_w_px : 1.f;
    const float lh = (leaf_h_px > 0.f && std::isfinite(leaf_h_px)) ? leaf_h_px : 1.f;

    const float hx_px = 0.5f * lw;
    const float hy_px = 0.5f * lh;

    const float dx_m = (d0 * inv_fx) * hx_px;
    const float dy_m = (d0 * inv_fy) * hy_px;

    float s = std::sqrt(dx_m * dx_m + dy_m * dy_m);
    s *= scale_mul_xy;

    if (!(s > 0.f) || !std::isfinite(s)) {
        s = 0.5f * (d0 * inv_fx);
    }
    const float min_abs = 1.0e-6f;
    if (!(s > min_abs) || !std::isfinite(s)) {
        s = min_abs;
    }

    const float zz = (z_scale > 0.f && std::isfinite(z_scale)) ? z_scale : 5.0e-6f;

    out->scale_xyz.x = s;
    out->scale_xyz.y = s;
    out->scale_xyz.z = zz;

    return true;
}

static inline bool make_oriented_gaussian_from_depth_pixel_precomp(OrientedGaussianInit* out,
                                                                   int u,
                                                                   int v,
                                                                   float leaf_w_px,
                                                                   float leaf_h_px,
                                                                   const float* depth,
                                                                   int W,
                                                                   int H,
                                                                   float near_plane,
                                                                   float fx,
                                                                   float fy,
                                                                   float cx,
                                                                   float cy,
                                                                   const Eigen::Matrix3f& R_WC,
                                                                   const Eigen::Vector3f& t_WC,
                                                                   const Eigen::Vector3f& exW,
                                                                   const Eigen::Vector3f& eyW,
                                                                   float scale_mul_xy,
                                                                   float z_scale)
{
    if (!out || !depth || W <= 0 || H <= 0) {
        return false;
    }

    u = std::clamp(u, 0, W - 1);
    v = std::clamp(v, 0, H - 1);
    const float d0 = depth[u + W * v];
    if (!(d0 > near_plane) || !std::isfinite(d0)) {
        return false;
    }

    const float inv_fx = 1.0f / fx;
    const float inv_fy = 1.0f / fy;
    gs::Point pos_w;
    if (!make_gaussian_center_from_valid_depth_pixel_precomp(
            &pos_w, u, v, d0, inv_fx, inv_fy, cx, cy, R_WC, t_WC)) {
        return false;
    }

    return finish_oriented_gaussian_from_depth_pixel_precomp(out,
                                                              pos_w,
                                                              u,
                                                              v,
                                                              leaf_w_px,
                                                              leaf_h_px,
                                                              d0,
                                                              depth,
                                                              W,
                                                              H,
                                                              near_plane,
                                                              inv_fx,
                                                              inv_fy,
                                                              cx,
                                                              cy,
                                                              R_WC,
                                                              exW,
                                                              eyW,
                                                              scale_mul_xy,
                                                              z_scale);
}

static inline bool make_oriented_gaussian_from_depth_pixel(OrientedGaussianInit* out,
                                                           int u,
                                                           int v,
                                                           int leaf_w,
                                                           int leaf_h,
                                                           const float* depth,
                                                           int W,
                                                           int H,
                                                           float near_plane,
                                                           float fx,
                                                           float fy,
                                                           float cx,
                                                           float cy,
                                                           const Eigen::Isometry3f& T_WC,
                                                           float scale_mul_xy,
                                                           float z_scale)
{
    const Eigen::Matrix3f R_WC = T_WC.linear();
    const Eigen::Vector3f t_WC = T_WC.translation();
    const Eigen::Vector3f exW = R_WC.col(0);
    const Eigen::Vector3f eyW = R_WC.col(1);

    return make_oriented_gaussian_from_depth_pixel_precomp(out,
                                                           u,
                                                           v,
                                                           leaf_w,
                                                           leaf_h,
                                                           depth,
                                                           W,
                                                           H,
                                                           near_plane,
                                                           fx,
                                                           fy,
                                                           cx,
                                                           cy,
                                                           R_WC,
                                                           t_WC,
                                                           exW,
                                                           eyW,
                                                           scale_mul_xy,
                                                           z_scale);
}

} 
