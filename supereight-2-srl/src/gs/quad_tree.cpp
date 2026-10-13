/*
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "gs/nvtx_utils.hpp"
#include "gs/quad_tree.cuh"

namespace gs {

cv::Mat Node::getPixels(const cv::Mat& img) const
{
    cv::Rect roi(x0_, y0_, width_, height_);
    return img(roi);
}

float Node::computeError(const cv::Mat& img) const
{
    cv::Mat pixels = getPixels(img);
    cv::Scalar avg_color = cv::mean(pixels);

    float mse0 = 0.0f, mse1 = 0.0f, mse2 = 0.0f;

    for (int i = 0; i < pixels.rows; ++i) {
        const cv::Vec3b* row = pixels.ptr<cv::Vec3b>(i);
        for (int j = 0; j < pixels.cols; ++j) {
            const cv::Vec3b& p = row[j];
            float d0 = static_cast<float>(p[0]) - static_cast<float>(avg_color[0]);
            float d1 = static_cast<float>(p[1]) - static_cast<float>(avg_color[1]);
            float d2 = static_cast<float>(p[2]) - static_cast<float>(avg_color[2]);
            mse0 += d0 * d0;
            mse1 += d1 * d1;
            mse2 += d2 * d2;
        }
    }

    const int count = pixels.rows * pixels.cols;
    mse0 /= count;
    mse1 /= count;
    mse2 /= count;

    const float err = mse0 * 0.2989f + mse1 * 0.5870f + mse2 * 0.1140f;
    return err * img.rows * img.cols / 90000000.0f;
}

static constexpr std::array<uint16_t, 256> make_sq_lut()
{
    std::array<uint16_t, 256> lut{};
    for (size_t i = 0; i < 256; ++i) lut[i] = static_cast<uint16_t>(i * i);
    return lut;
}
static constexpr auto kSqLUT = make_sq_lut();

static int effective_parallel_threads(const int requested_threads,
                                      const size_t work_items,
                                      const size_t min_items_per_thread)
{
    const int requested = std::max(1, requested_threads);
    if (requested == 1 || work_items < 2 * min_items_per_thread) {
        return 1;
    }
    const size_t useful = (work_items + min_items_per_thread - 1) / min_items_per_thread;
    return std::max(1, std::min(requested, static_cast<int>(useful)));
}

void build_integral_sums_rgb(const cv::Mat& img,
                             IntegralSumsRGB& out,
                             bool build_sumsq,
                             int thread_count)
{
    out.rows = img.rows;
    out.cols = img.cols;
    out.stride = out.cols + 1;

    const size_t N = static_cast<size_t>(out.stride) * static_cast<size_t>(out.rows + 1);

    if (out.sum.size() != N) out.sum.resize(N);
    if (build_sumsq) {
        if (out.sq.size() != N) out.sq.resize(N);
    } else {
        out.sq.clear(); 
    }

    const int stride = out.stride;
    constexpr size_t kMinPixelsPerScanThread = 128 * 1024;
    const size_t pixel_count = static_cast<size_t>(out.rows) * static_cast<size_t>(out.cols);
    thread_count = effective_parallel_threads(
        thread_count, pixel_count, kMinPixelsPerScanThread);

    for (int x = 0; x < stride; ++x) {
        out.sum[x].s0 = 0u;
        out.sum[x].s1 = 0u;
        out.sum[x].s2 = 0u;
        if (build_sumsq) {
            out.sq[x].q0 = 0ull;
            out.sq[x].q1 = 0ull;
            out.sq[x].q2 = 0ull;
        }
    }

    if (thread_count == 1) {
        for (int y = 1; y <= out.rows; ++y) {
            const cv::Vec3b* row = img.ptr<cv::Vec3b>(y - 1);

            uint32_t run0 = 0, run1 = 0, run2 = 0;
            uint64_t run0sq = 0, run1sq = 0, run2sq = 0;

            Sum3* sum_row = out.sum.data() + y * stride;
            const Sum3* sum_prev = sum_row - stride;
            sum_row[0] = Sum3{0u, 0u, 0u};

            Sq3* sq_row = build_sumsq ? out.sq.data() + y * stride : nullptr;
            const Sq3* sq_prev = build_sumsq ? sq_row - stride : nullptr;
            if (sq_row) {
                sq_row[0] = Sq3{0ull, 0ull, 0ull};
            }

            for (int x = 1; x <= out.cols; ++x) {
                const cv::Vec3b& p = row[x - 1];
                run0 += static_cast<uint32_t>(p[0]);
                run1 += static_cast<uint32_t>(p[1]);
                run2 += static_cast<uint32_t>(p[2]);
                sum_row[x] = Sum3{static_cast<uint32_t>(sum_prev[x].s0 + run0),
                                  static_cast<uint32_t>(sum_prev[x].s1 + run1),
                                  static_cast<uint32_t>(sum_prev[x].s2 + run2)};

                if (sq_row) {
                    run0sq += static_cast<uint64_t>(kSqLUT[p[0]]);
                    run1sq += static_cast<uint64_t>(kSqLUT[p[1]]);
                    run2sq += static_cast<uint64_t>(kSqLUT[p[2]]);
                    sq_row[x] = Sq3{sq_prev[x].q0 + run0sq,
                                    sq_prev[x].q1 + run1sq,
                                    sq_prev[x].q2 + run2sq};
                }
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (int y = 1; y <= out.rows; ++y) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y - 1);

        uint32_t run0 = 0, run1 = 0, run2 = 0;
        uint64_t run0sq = 0, run1sq = 0, run2sq = 0;

        Sum3*       sum_row  = out.sum.data() + y * stride;
        sum_row[0] = Sum3{0u, 0u, 0u};

        Sq3*       sq_row  = nullptr;
        if (build_sumsq) {
            sq_row  = out.sq.data() + y * stride;
            sq_row[0] = Sq3{0ull, 0ull, 0ull};
        }

        for (int x = 1; x <= out.cols; ++x) {
            const cv::Vec3b& p = row[x - 1];

            run0 += static_cast<uint32_t>(p[0]);
            run1 += static_cast<uint32_t>(p[1]);
            run2 += static_cast<uint32_t>(p[2]);

            sum_row[x] = Sum3{run0, run1, run2};

            if (build_sumsq) {
                run0sq += static_cast<uint64_t>(kSqLUT[p[0]]);
                run1sq += static_cast<uint64_t>(kSqLUT[p[1]]);
                run2sq += static_cast<uint64_t>(kSqLUT[p[2]]);

                sq_row[x] = Sq3{run0sq, run1sq, run2sq};
            }
        }
    }

    constexpr int kColumnTile = 64;
    const int tile_count = (out.cols + kColumnTile - 1) / kColumnTile;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (int tile = 0; tile < tile_count; ++tile) {
        const int x_begin = 1 + tile * kColumnTile;
        const int x_end = std::min(out.cols + 1, x_begin + kColumnTile);
        for (int y = 1; y <= out.rows; ++y) {
            Sum3* row = out.sum.data() + y * stride;
            const Sum3* prev = row - stride;
            for (int x = x_begin; x < x_end; ++x) {
                row[x].s0 += prev[x].s0;
                row[x].s1 += prev[x].s1;
                row[x].s2 += prev[x].s2;
            }

            if (build_sumsq) {
                Sq3* sq_row = out.sq.data() + y * stride;
                const Sq3* sq_prev = sq_row - stride;
                for (int x = x_begin; x < x_end; ++x) {
                    sq_row[x].q0 += sq_prev[x].q0;
                    sq_row[x].q1 += sq_prev[x].q1;
                    sq_row[x].q2 += sq_prev[x].q2;
                }
            }
        }
    }
}

void build_leaf_id_lut(const std::vector<Leaf>& leaves,
                       const int width,
                       const int height,
                       std::vector<int32_t>& leaf_ids,
                       int thread_count)
{
    if (width <= 0 || height <= 0) {
        leaf_ids.clear();
        return;
    }

    leaf_ids.assign(static_cast<size_t>(width) * static_cast<size_t>(height), int32_t(-1));
    int32_t* const leaf_id_data = leaf_ids.data();
    const Leaf* const leaf_data = leaves.data();
    const int leaf_count = static_cast<int>(leaves.size());
    constexpr size_t kMinPixelsPerLutThread = 128 * 1024;
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    thread_count = effective_parallel_threads(
        thread_count, pixel_count, kMinPixelsPerLutThread);
    if (leaf_count < 2 * thread_count) {
        thread_count = 1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(thread_count > 1)
#endif
    for (int lid = 0; lid < leaf_count; ++lid) {
        const Leaf& leaf = leaf_data[static_cast<size_t>(lid)];
        if (leaf.w <= 0 || leaf.h <= 0) {
            continue;
        }

        const int x_begin = std::max(0, leaf.x);
        const int y_begin = std::max(0, leaf.y);
        const int x_end = std::min(width, leaf.x + leaf.w);
        const int y_end = std::min(height, leaf.y + leaf.h);
        if (x_begin >= x_end || y_begin >= y_end) {
            continue;
        }

        for (int y = y_begin; y < y_end; ++y) {
            int32_t* const row = leaf_id_data + static_cast<size_t>(y) * width;
            std::fill(row + x_begin, row + x_end, static_cast<int32_t>(lid));
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
#define GS_RESTRICT __restrict__
#else
#define GS_RESTRICT
#endif

struct IntegralView {
    int stride = 0;
    const Sum3* GS_RESTRICT sum = nullptr;
    const Sq3*  GS_RESTRICT sq  = nullptr; 
};

static inline IntegralView make_view(const IntegralSumsRGB& s)
{
    IntegralView v;
    v.stride = s.stride;
    v.sum = s.sum.empty() ? nullptr : s.sum.data();
    v.sq  = s.sq.empty()  ? nullptr : s.sq.data();
    return v;
}

static inline void rect_sums3_o1_corners_view(const IntegralView& v,
                                             int x0, int y0, int w, int h,
                                             uint64_t& s0, uint64_t& s1, uint64_t& s2)
{
    const int stride = v.stride;
    const int x1 = x0 + w;

    const Sum3* GS_RESTRICT row0 = v.sum + y0 * stride;
    const Sum3* GS_RESTRICT row1 = row0 + h * stride;

    const Sum3* GS_RESTRICT p00 = row0 + x0;
    const Sum3* GS_RESTRICT p01 = row0 + x1;
    const Sum3* GS_RESTRICT p10 = row1 + x0;
    const Sum3* GS_RESTRICT p11 = row1 + x1;

    s0 = static_cast<uint64_t>(p11->s0) - static_cast<uint64_t>(p01->s0)
       - static_cast<uint64_t>(p10->s0) + static_cast<uint64_t>(p00->s0);
    s1 = static_cast<uint64_t>(p11->s1) - static_cast<uint64_t>(p01->s1)
       - static_cast<uint64_t>(p10->s1) + static_cast<uint64_t>(p00->s1);
    s2 = static_cast<uint64_t>(p11->s2) - static_cast<uint64_t>(p01->s2)
       - static_cast<uint64_t>(p10->s2) + static_cast<uint64_t>(p00->s2);
}

static inline void rect_sums6_o1_corners_view(const IntegralView& v,
                                             int x0, int y0, int w, int h,
                                             uint64_t& s0, uint64_t& s1, uint64_t& s2,
                                             uint64_t& ss0, uint64_t& ss1, uint64_t& ss2)
{
    const int stride = v.stride;
    const int x1 = x0 + w;

    const Sum3* GS_RESTRICT srow0 = v.sum + y0 * stride;
    const Sum3* GS_RESTRICT srow1 = srow0 + h * stride;

    const Sum3* GS_RESTRICT s00 = srow0 + x0;
    const Sum3* GS_RESTRICT s01 = srow0 + x1;
    const Sum3* GS_RESTRICT s10 = srow1 + x0;
    const Sum3* GS_RESTRICT s11 = srow1 + x1;

    s0 = static_cast<uint64_t>(s11->s0) - static_cast<uint64_t>(s01->s0)
       - static_cast<uint64_t>(s10->s0) + static_cast<uint64_t>(s00->s0);
    s1 = static_cast<uint64_t>(s11->s1) - static_cast<uint64_t>(s01->s1)
       - static_cast<uint64_t>(s10->s1) + static_cast<uint64_t>(s00->s1);
    s2 = static_cast<uint64_t>(s11->s2) - static_cast<uint64_t>(s01->s2)
       - static_cast<uint64_t>(s10->s2) + static_cast<uint64_t>(s00->s2);

    const Sq3* GS_RESTRICT qrow0 = v.sq + y0 * stride;
    const Sq3* GS_RESTRICT qrow1 = qrow0 + h * stride;

    const Sq3* GS_RESTRICT q00 = qrow0 + x0;
    const Sq3* GS_RESTRICT q01 = qrow0 + x1;
    const Sq3* GS_RESTRICT q10 = qrow1 + x0;
    const Sq3* GS_RESTRICT q11 = qrow1 + x1;

    ss0 = q11->q0 - q01->q0 - q10->q0 + q00->q0;
    ss1 = q11->q1 - q01->q1 - q10->q1 + q00->q1;
    ss2 = q11->q2 - q01->q2 - q10->q2 + q00->q2;
}

static inline float compute_img_scale(const cv::Mat& img)
{
    const double scale = (static_cast<double>(img.rows) * static_cast<double>(img.cols)) / 90000000.0;
    return static_cast<float>(scale);
}

static inline float compute_error_impl4_o1_node(const Node& node, const IntegralView& v, float img_scale)
{
    const int x0 = node.getOriginX();
    const int y0 = node.getOriginY();
    const int w  = node.getWidth();
    const int h  = node.getHeight();

    const int count = w * h;
    const double inv_count = 1.0 / static_cast<double>(count);

    uint64_t s0, s1, s2, ss0, ss1, ss2;
    rect_sums6_o1_corners_view(v, x0, y0, w, h, s0, s1, s2, ss0, ss1, ss2);

    const double m0   = static_cast<double>(s0)  * inv_count;
    const double m1   = static_cast<double>(s1)  * inv_count;
    const double m2   = static_cast<double>(s2)  * inv_count;

    const double ex20 = static_cast<double>(ss0) * inv_count;
    const double ex21 = static_cast<double>(ss1) * inv_count;
    const double ex22 = static_cast<double>(ss2) * inv_count;

    double mse0 = ex20 - m0 * m0;
    double mse1 = ex21 - m1 * m1;
    double mse2 = ex22 - m2 * m2;

    if (mse0 < 0.0) mse0 = 0.0;
    if (mse1 < 0.0) mse1 = 0.0;
    if (mse2 < 0.0) mse2 = 0.0;

    const float err = static_cast<float>(mse0 * 0.2989 + mse1 * 0.5870 + mse2 * 0.1140);
    return err * img_scale;
}

using Rect = Leaf;

static inline int gs_clampi_int(const int v, const int lo, const int hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline float compute_error_integral_sumsq_o1_rect_view_with_sums(const Rect& r,
                                                                        const IntegralView& v,
                                                                        float img_scale,
                                                                        uint64_t& s0, uint64_t& s1, uint64_t& s2)
{
    const int count = r.w * r.h;
    const double inv_count = 1.0 / static_cast<double>(count);

    uint64_t ss0, ss1, ss2;
    rect_sums6_o1_corners_view(v, r.x, r.y, r.w, r.h, s0, s1, s2, ss0, ss1, ss2);

    const double m0   = static_cast<double>(s0)  * inv_count;
    const double m1   = static_cast<double>(s1)  * inv_count;
    const double m2   = static_cast<double>(s2)  * inv_count;

    const double ex20 = static_cast<double>(ss0) * inv_count;
    const double ex21 = static_cast<double>(ss1) * inv_count;
    const double ex22 = static_cast<double>(ss2) * inv_count;

    double mse0 = ex20 - m0 * m0;
    double mse1 = ex21 - m1 * m1;
    double mse2 = ex22 - m2 * m2;

    if (mse0 < 0.0) mse0 = 0.0;
    if (mse1 < 0.0) mse1 = 0.0;
    if (mse2 < 0.0) mse2 = 0.0;

    const float err = static_cast<float>(mse0 * 0.2989 + mse1 * 0.5870 + mse2 * 0.1140);
    return err * img_scale;
}

static inline float compute_ring_contrast_luma_sq_scaled_rect_view(const Rect& r,
                                                                   const IntegralView& v,
                                                                   const int img_w,
                                                                   const int img_h,
                                                                   const int ring_t,
                                                                   const float img_scale,
                                                                   const uint64_t s0r,
                                                                   const uint64_t s1r,
                                                                   const uint64_t s2r)
{
    if (ring_t <= 0) return 0.0f;

    const int x0e = gs_clampi_int(r.x - ring_t, 0, img_w);
    const int y0e = gs_clampi_int(r.y - ring_t, 0, img_h);
    const int x1e = gs_clampi_int(r.x + r.w + ring_t, 0, img_w);
    const int y1e = gs_clampi_int(r.y + r.h + ring_t, 0, img_h);

    const int wE = x1e - x0e;
    const int hE = y1e - y0e;
    if (wE <= 0 || hE <= 0) return 0.0f;

    const uint64_t cntR = static_cast<uint64_t>(r.w) * static_cast<uint64_t>(r.h);
    const uint64_t cntE = static_cast<uint64_t>(wE) * static_cast<uint64_t>(hE);
    if (cntR == 0 || cntE <= cntR) return 0.0f;

    uint64_t s0e, s1e, s2e;
    rect_sums3_o1_corners_view(v, x0e, y0e, wE, hE, s0e, s1e, s2e);

    const uint64_t sr0 = s0e - s0r;
    const uint64_t sr1 = s1e - s1r;
    const uint64_t sr2 = s2e - s2r;

    const uint64_t cntRing = cntE - cntR;
    if (cntRing == 0) return 0.0f;

    const double invR    = 1.0 / static_cast<double>(cntR);
    const double invRing = 1.0 / static_cast<double>(cntRing);

    const double m0R = static_cast<double>(s0r) * invR;
    const double m1R = static_cast<double>(s1r) * invR;
    const double m2R = static_cast<double>(s2r) * invR;

    const double m0Ring = static_cast<double>(sr0) * invRing;
    const double m1Ring = static_cast<double>(sr1) * invRing;
    const double m2Ring = static_cast<double>(sr2) * invRing;

    const float yR    = static_cast<float>(m0R    * 0.2989 + m1R    * 0.5870 + m2R    * 0.1140);
    const float yRing = static_cast<float>(m0Ring * 0.2989 + m1Ring * 0.5870 + m2Ring * 0.1140);

    const float d = yRing - yR;
    return (d * d) * img_scale;
}

static inline bool rect_should_split(const Rect& rect,
                                     float threshold,
                                     float ring_threshold,
                                     int ring_t,
                                     int min_pixel_size,
                                     const IntegralView& v,
                                     float img_scale,
                                     int img_w,
                                     int img_h)
{
    uint64_t s0r = 0, s1r = 0, s2r = 0;
    const float err = compute_error_integral_sumsq_o1_rect_view_with_sums(
        rect, v, img_scale, s0r, s1r, s2r);
    const int w1 = rect.w >> 1;
    const int h1 = rect.h >> 1;
    if (w1 <= min_pixel_size || h1 <= min_pixel_size) {
        return false;
    }
    if (err > threshold) {
        return true;
    }
    if (ring_t <= 0) {
        return false;
    }
    const float ring_contrast = compute_ring_contrast_luma_sq_scaled_rect_view(
        rect, v, img_w, img_h, ring_t, img_scale, s0r, s1r, s2r);
    return ring_contrast > ring_threshold;
}

static inline void push_children_reverse(const Rect& rect, std::vector<Rect>& stack)
{
    const int w1 = rect.w >> 1;
    const int w2 = rect.w - w1;
    const int h1 = rect.h >> 1;
    const int h2 = rect.h - h1;
    stack.push_back(Rect{rect.x + w1, rect.y + h1, w2, h2});
    stack.push_back(Rect{rect.x + w1, rect.y, w2, h1});
    stack.push_back(Rect{rect.x, rect.y + h1, w1, h2});
    stack.push_back(Rect{rect.x, rect.y, w1, h1});
}

static void iterative_subdivide_impl6_rect_stack(const Rect& root,
                                                 float threshold,
                                                 float ring_threshold,
                                                 int ring_t,
                                                 int min_pixel_size,
                                                 std::vector<Leaf>& out_leaves,
                                                 const IntegralView& v,
                                                 float img_scale,
                                                 int img_w,
                                                 int img_h)
{
    static thread_local std::vector<Rect> st;
    st.clear();
    if (st.capacity() < 1024) {
        st.reserve(1024);
    }
    st.push_back(root);

    while (!st.empty()) {
        const Rect cur = st.back();
        st.pop_back();
        if (!rect_should_split(cur,
                               threshold,
                               ring_threshold,
                               ring_t,
                               min_pixel_size,
                               v,
                               img_scale,
                               img_w,
                               img_h)) {
            out_leaves.push_back(cur);
            continue;
        }
        push_children_reverse(cur, st);
    }
}

static void iterative_subdivide_impl4_rect_stack(const Rect& root,
                                                 float threshold,
                                                 int min_pixel_size,
                                                 std::vector<Leaf>& out_leaves,
                                                 const IntegralView& v,
                                                 float img_scale)
{
    static thread_local std::vector<Rect> st;
    st.clear();
    if (st.capacity() < 1024) {
        st.reserve(1024);
    }
    st.push_back(root);

    while (!st.empty()) {
        const Rect cur = st.back();
        st.pop_back();

        if (!rect_should_split(cur,
                               threshold,
                               threshold,
                               0,
                               min_pixel_size,
                               v,
                               img_scale,
                               root.w,
                               root.h)) {
            out_leaves.push_back(cur);
            continue;
        }
        push_children_reverse(cur, st);
    }
}

static void build_parallel_frontier(const Rect& rect,
                                    int depth,
                                    int split_depth,
                                    float threshold,
                                    float ring_threshold,
                                    int ring_t,
                                    int min_pixel_size,
                                    const IntegralView& view,
                                    float img_scale,
                                    int img_w,
                                    int img_h,
                                    std::vector<Leaf>& frontier,
                                    std::vector<uint8_t>& is_leaf)
{
    if (!rect_should_split(rect,
                           threshold,
                           ring_threshold,
                           ring_t,
                           min_pixel_size,
                           view,
                           img_scale,
                           img_w,
                           img_h)) {
        frontier.push_back(rect);
        is_leaf.push_back(uint8_t(1));
        return;
    }
    if (depth >= split_depth) {
        frontier.push_back(rect);
        is_leaf.push_back(uint8_t(0));
        return;
    }

    const int w1 = rect.w >> 1;
    const int w2 = rect.w - w1;
    const int h1 = rect.h >> 1;
    const int h2 = rect.h - h1;
    build_parallel_frontier(Rect{rect.x, rect.y, w1, h1},
                            depth + 1,
                            split_depth,
                            threshold,
                            ring_threshold,
                            ring_t,
                            min_pixel_size,
                            view,
                            img_scale,
                            img_w,
                            img_h,
                            frontier,
                            is_leaf);
    build_parallel_frontier(Rect{rect.x, rect.y + h1, w1, h2},
                            depth + 1,
                            split_depth,
                            threshold,
                            ring_threshold,
                            ring_t,
                            min_pixel_size,
                            view,
                            img_scale,
                            img_w,
                            img_h,
                            frontier,
                            is_leaf);
    build_parallel_frontier(Rect{rect.x + w1, rect.y, w2, h1},
                            depth + 1,
                            split_depth,
                            threshold,
                            ring_threshold,
                            ring_t,
                            min_pixel_size,
                            view,
                            img_scale,
                            img_w,
                            img_h,
                            frontier,
                            is_leaf);
    build_parallel_frontier(Rect{rect.x + w1, rect.y + h1, w2, h2},
                            depth + 1,
                            split_depth,
                            threshold,
                            ring_threshold,
                            ring_t,
                            min_pixel_size,
                            view,
                            img_scale,
                            img_w,
                            img_h,
                            frontier,
                            is_leaf);
}

static void recursive_subdivide_impl4(Node& node,
                                      float threshold,
                                      int min_pixel_size,
                                      const IntegralView& v,
                                      float img_scale,
                                      std::vector<Leaf>& out_leaves,
                                      bool build_tree)
{
    const float err = compute_error_impl4_o1_node(node, v, img_scale);

    if (err <= threshold) {
        out_leaves.emplace_back(
            node.getOriginX(), node.getOriginY(), node.getWidth(), node.getHeight());
        return;
    }

    const int w = node.getWidth();
    const int h = node.getHeight();
    const int w1 = w >> 1;
    const int w2 = w - w1;
    const int h1 = h >> 1;
    const int h2 = h - h1;

    if (w1 <= min_pixel_size || h1 <= min_pixel_size) {
        out_leaves.emplace_back(
            node.getOriginX(), node.getOriginY(), node.getWidth(), node.getHeight());
        return;
    }

    Node n1(node.getOriginX(),         node.getOriginY(),         w1, h1);
    recursive_subdivide_impl4(n1, threshold, min_pixel_size, v, img_scale, out_leaves, build_tree);

    Node n2(node.getOriginX(),         node.getOriginY() + h1,    w1, h2);
    recursive_subdivide_impl4(n2, threshold, min_pixel_size, v, img_scale, out_leaves, build_tree);

    Node n3(node.getOriginX() + w1,    node.getOriginY(),         w2, h1);
    recursive_subdivide_impl4(n3, threshold, min_pixel_size, v, img_scale, out_leaves, build_tree);

    Node n4(node.getOriginX() + w1,    node.getOriginY() + h1,    w2, h2);
    recursive_subdivide_impl4(n4, threshold, min_pixel_size, v, img_scale, out_leaves, build_tree);

    if (build_tree) {
        node.children.clear();
        node.children.reserve(4);
        node.children.push_back(std::move(n1));
        node.children.push_back(std::move(n2));
        node.children.push_back(std::move(n3));
        node.children.push_back(std::move(n4));
    }
}

void QTree::subdivide(bool build_tree)
{
    GOFUSION_NVTX_RANGE("GOFusion/QTree/subdivide");
    if (!img_) return;

    const size_t previous_leaf_count = leaves_.size();
    leaves_.clear();
    root_.children.clear();

    {
        GOFUSION_NVTX_RANGE("GOFusion/QTree/integral_scan");
        build_integral_sums_rgb(*img_, sums_, true, thread_count_);
    }
    const IntegralView v = make_view(sums_);
    const float img_scale = compute_img_scale(*img_);

    if (leaves_.capacity() == 0) {
        leaves_.reserve(std::max<size_t>(
            1024, static_cast<size_t>(img_->rows) * static_cast<size_t>(img_->cols) / 32));
    }
    else if (previous_leaf_count > leaves_.capacity() * 9 / 10) {
        leaves_.reserve(previous_leaf_count + previous_leaf_count / 4);
    }

    const float ring_threshold = (ring_threshold_ >= 0.0f) ? ring_threshold_ : threshold_;
    const int   ring_t         = (ring_t_ > 0) ? ring_t_ : 0;

    GOFUSION_NVTX_RANGE("GOFusion/QTree/ordered_dfs");
    if (!build_tree && v.sum && v.sq && v.stride > 0) {
        const Rect root{root_.getOriginX(), root_.getOriginY(), root_.getWidth(), root_.getHeight()};
        if (thread_count_ <= 1 || split_depth_ <= 0) {
            if (ring_t <= 0) {
                iterative_subdivide_impl4_rect_stack(
                    root, threshold_, min_pixel_size_, leaves_, v, img_scale);
            }
            else {
                iterative_subdivide_impl6_rect_stack(root,
                                                     threshold_,
                                                     ring_threshold,
                                                     ring_t,
                                                     min_pixel_size_,
                                                     leaves_,
                                                     v,
                                                     img_scale,
                                                     img_->cols,
                                                     img_->rows);
            }
            return;
        }

        frontier_.clear();
        frontier_is_leaf_.clear();
        const size_t max_frontier = size_t(1) << (2 * split_depth_);
        if (frontier_.capacity() < max_frontier) {
            frontier_.reserve(max_frontier);
            frontier_is_leaf_.reserve(max_frontier);
        }
        build_parallel_frontier(root,
                                0,
                                split_depth_,
                                threshold_,
                                ring_threshold,
                                ring_t,
                                min_pixel_size_,
                                v,
                                img_scale,
                                img_->cols,
                                img_->rows,
                                frontier_,
                                frontier_is_leaf_);

        if (frontier_outputs_.size() < frontier_.size()) {
            frontier_outputs_.resize(frontier_.size());
        }
        size_t subtree_count = 0;
        for (const uint8_t is_leaf : frontier_is_leaf_) {
            subtree_count += is_leaf == 0;
        }
        const int dfs_threads = effective_parallel_threads(
            thread_count_, subtree_count, size_t(2));

        const auto process_frontier = [&](const size_t i) {
            auto& output = frontier_outputs_[i];
            output.clear();
            if (frontier_is_leaf_[i] != 0) {
                output.push_back(frontier_[i]);
            }
            else if (ring_t <= 0) {
                iterative_subdivide_impl4_rect_stack(frontier_[i],
                                                     threshold_,
                                                     min_pixel_size_,
                                                     output,
                                                     v,
                                                     img_scale);
            }
            else {
                iterative_subdivide_impl6_rect_stack(frontier_[i],
                                                     threshold_,
                                                     ring_threshold,
                                                     ring_t,
                                                     min_pixel_size_,
                                                     output,
                                                     v,
                                                     img_scale,
                                                     img_->cols,
                                                     img_->rows);
            }
        };

        if (dfs_threads <= 1) {
            for (size_t i = 0; i < frontier_.size(); ++i) {
                process_frontier(i);
            }
        }
        else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(dfs_threads)
#endif
            for (int i = 0; i < static_cast<int>(frontier_.size()); ++i) {
                process_frontier(static_cast<size_t>(i));
            }
        }

        size_t total_leaves = 0;
        for (size_t i = 0; i < frontier_.size(); ++i) {
            total_leaves += frontier_outputs_[i].size();
        }
        if (leaves_.capacity() < total_leaves) {
            leaves_.reserve(total_leaves);
        }
        for (size_t i = 0; i < frontier_.size(); ++i) {
            const auto& output = frontier_outputs_[i];
            leaves_.insert(leaves_.end(), output.begin(), output.end());
        }
        return;
    }

    recursive_subdivide_impl4(root_, threshold_, min_pixel_size_, v, img_scale, leaves_, build_tree);
}

void QTree::renderImg(int thickness, cv::Scalar color)
{
    if (!img_) return;
    if (img_->empty()) return;
    if (img_->type() != CV_8UC3) return;

    cv::Mat imgc;
    cv::cvtColor(*img_, imgc, cv::COLOR_RGB2BGR);
    cv::imshow("before", imgc);

    const std::vector<Leaf>* children_ptr = &leaves_;
    std::vector<Leaf> tmp_children;

    if (children_ptr->empty()) {
        const auto tree_children = find_children(root_);
        tmp_children.reserve(tree_children.size());
        for (const auto& child : tree_children) {
            tmp_children.emplace_back(child.getOriginX(),
                                      child.getOriginY(),
                                      child.getWidth(),
                                      child.getHeight());
        }
        children_ptr = &tmp_children;
    }

    const size_t needN = static_cast<size_t>(img_->cols + 1) * static_cast<size_t>(img_->rows + 1);
    const bool sums_ready =
        (sums_.rows == img_->rows) &&
        (sums_.cols == img_->cols) &&
        (sums_.stride == img_->cols + 1) &&
        (sums_.sum.size() == needN);

    if (!sums_ready) {
        build_integral_sums_rgb(*img_, sums_, false, thread_count_);
    }
    const IntegralView v = make_view(sums_);

    std::cout << "Find " << children_ptr->size() << " nodes" << std::endl;

    for (const auto& child : *children_ptr) {
        const int x0 = child.getOriginX();
        const int y0 = child.getOriginY();
        const int w  = child.getWidth();
        const int h  = child.getHeight();

        const uint64_t cnt = static_cast<uint64_t>(w) * static_cast<uint64_t>(h);
        if (cnt == 0) continue;

        uint64_t s0, s1, s2;
        rect_sums3_o1_corners_view(v, x0, y0, w, h, s0, s1, s2);

        const int m0 = static_cast<int>(s0 / cnt);
        const int m1 = static_cast<int>(s1 / cnt);
        const int m2 = static_cast<int>(s2 / cnt);

        const int avg_b = m0;
        const int avg_g = m1;
        const int avg_r = m2;

        imgc(cv::Rect(x0, y0, w, h)).setTo(cv::Scalar(avg_r, avg_g, avg_b));

        if (thickness > 0) {
            cv::rectangle(imgc,
                          cv::Point(x0, y0),
                          cv::Point(x0 + w, y0 + h),
                          color,
                          thickness);
        }
    }

    cv::imshow("after", imgc);
    cv::waitKey(0);
}

QTreeWorker::QTreeWorker() : worker_(&QTreeWorker::run, this) {}

QTreeWorker::~QTreeWorker()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void QTreeWorker::submit(const cv::Mat& image,
                         float threshold,
                         int min_pixel_size,
                         float ring_threshold,
                         int ring_t,
                         int thread_count,
                         int split_depth,
                         std::shared_ptr<void> source_owner)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_ || !complete_) {
        throw std::logic_error("QTreeWorker submit called before the previous wait completed");
    }
    image_ = image;
    threshold_ = threshold;
    min_pixel_size_ = min_pixel_size;
    ring_threshold_ = ring_threshold;
    ring_t_ = ring_t;
    thread_count_ = std::max(1, thread_count);
    split_depth_ = std::clamp(split_depth, 0, 4);
    source_owner_ = std::move(source_owner);
    error_ = nullptr;
    complete_ = false;
    pending_ = true;
    work_cv_.notify_one();
}

const std::vector<Leaf>& QTreeWorker::wait()
{
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return complete_; });
    if (error_) {
        std::rethrow_exception(error_);
    }
    if (!qtree_) {
        throw std::logic_error("QTreeWorker completed without a quadtree result");
    }
    return qtree_->getAllLeaves();
}

void QTreeWorker::run()
{
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this]() { return stop_ || pending_; });
            if (stop_) {
                return;
            }
        }

        std::exception_ptr error;
        try {
            if (!qtree_) {
                qtree_ = std::make_unique<QTree>(threshold_, min_pixel_size_, image_);
            }
            else {
                qtree_->setParams(threshold_, min_pixel_size_);
                qtree_->setImage(image_);
            }
            qtree_->setRingParams(ring_threshold_, ring_t_);
            qtree_->setExecutionParams(thread_count_, split_depth_);
            qtree_->subdivide(false);
        }
        catch (...) {
            error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = error;
            source_owner_.reset();
            pending_ = false;
            complete_ = true;
        }
        done_cv_.notify_one();
    }
}

void recursive_subdivide(Node& node, float threshold, int min_pixel_size, cv::Mat& img)
{
    if (node.computeError(img) <= threshold) {
        return;
    }

    int w1 = static_cast<int>(std::floor(node.getWidth() / 2.0));
    int w2 = static_cast<int>(std::ceil(node.getWidth() / 2.0));
    int h1 = static_cast<int>(std::floor(node.getHeight() / 2.0));
    int h2 = static_cast<int>(std::ceil(node.getHeight() / 2.0));

    if (w1 <= min_pixel_size || h1 <= min_pixel_size) {
        return;
    }

    Node n1(node.getOriginX(), node.getOriginY(), w1, h1);
    recursive_subdivide(n1, threshold, min_pixel_size, img);

    Node n2(node.getOriginX(), node.getOriginY() + h1, w1, h2);
    recursive_subdivide(n2, threshold, min_pixel_size, img);

    Node n3(node.getOriginX() + w1, node.getOriginY(), w2, h1);
    recursive_subdivide(n3, threshold, min_pixel_size, img);

    Node n4(node.getOriginX() + w1, node.getOriginY() + h1, w2, h2);
    recursive_subdivide(n4, threshold, min_pixel_size, img);

    std::vector<Node> children{n1, n2, n3, n4};
    node.children = children;
}

std::vector<Node> find_children(const Node& node)
{
    if (node.children.empty()) {
        return {node};
    }

    std::vector<Node> all_children;
    for (const auto& child : node.children) {
        auto grandchildren = find_children(child);
        all_children.insert(all_children.end(), grandchildren.begin(), grandchildren.end());
    }
    return all_children;
}

} 
