/*
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

#ifndef GS_QUAD_TREE_HPP
#define GS_QUAD_TREE_HPP

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>
#include <opencv2/opencv.hpp>

namespace gs {

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two");

    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n)
    {
        if (n == 0) return nullptr;

        void* p = nullptr;

#if defined(_MSC_VER)
        p = _aligned_malloc(n * sizeof(T), Alignment);
        if (!p) throw std::bad_alloc();
#else
        if (::posix_memalign(&p, Alignment, n * sizeof(T)) != 0) p = nullptr;
        if (!p) throw std::bad_alloc();
#endif
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <class T1, std::size_t A1, class T2, std::size_t A2>
constexpr bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept
{
    return A1 == A2;
}
template <class T1, std::size_t A1, class T2, std::size_t A2>
constexpr bool operator!=(const AlignedAllocator<T1, A1>& a, const AlignedAllocator<T2, A2>& b) noexcept
{
    return !(a == b);
}

struct Sum3 {
    uint32_t s0, s1, s2;
};

struct Sq3 {
    uint64_t q0, q1, q2;
};

struct IntegralSumsRGB {
    int rows = 0;
    int cols = 0;
    int stride = 0; 

    std::vector<Sum3, AlignedAllocator<Sum3, 64>> sum; 
    std::vector<Sq3,  AlignedAllocator<Sq3,  64>> sq;  
};

struct Leaf {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    constexpr Leaf() = default;
    constexpr Leaf(int x0, int y0, int width, int height) : x(x0), y(y0), w(width), h(height) {}

    inline int getOriginX() const { return x; }
    inline int getOriginY() const { return y; }
    inline int getWidth() const { return w; }
    inline int getHeight() const { return h; }
};

static_assert(sizeof(Leaf) == 4 * sizeof(int), "Leaf must remain a compact POD rectangle");
static_assert(std::is_standard_layout_v<Leaf> && std::is_trivially_copyable_v<Leaf>,
              "Leaf must remain POD-like");

class Node {
public:
    Node(int x0, int y0, int width, int height)
        : x0_(x0), y0_(y0), width_(width), height_(height)
    {}

    std::vector<Node> children;

    inline int getOriginX() const { return x0_; }
    inline int getOriginY() const { return y0_; }
    inline int getWidth() const { return width_; }
    inline int getHeight() const { return height_; }

    inline void resetBounds(int x0, int y0, int w, int h)
    {
        x0_ = x0;
        y0_ = y0;
        width_ = w;
        height_ = h;
        children.clear();
    }

    cv::Mat getPixels(const cv::Mat& img) const;

    float computeError(const cv::Mat& img) const;

private:
    int x0_;
    int y0_;
    int width_;
    int height_;
};

class QTree {
public:
    QTree(float threshold, int min_pixel_size, cv::Mat& img)
        : threshold_(threshold)
        , min_pixel_size_(min_pixel_size)
        , img_(&img)
        , root_(0, 0, img.cols, img.rows)
    {}

    inline void setImage(cv::Mat& img)
    {
        img_ = &img;
        root_.resetBounds(0, 0, img.cols, img.rows);
    }

    inline void setParams(float threshold, int min_pixel_size)
    {
        threshold_ = threshold;
        min_pixel_size_ = min_pixel_size;
    }

    inline void setRingParams(float ring_threshold, int ring_t)
    {
        ring_threshold_ = ring_threshold;
        ring_t_ = ring_t;
    }

    inline const std::vector<Leaf>& getAllLeaves() const { return leaves_; }
    inline const Leaf& getLeaf(const int n) const { return leaves_[n]; }

    inline void setExecutionParams(int thread_count, int split_depth)
    {
        thread_count_ = std::max(1, thread_count);
        split_depth_ = std::clamp(split_depth, 0, 4);
    }

    void subdivide(bool build_tree = false);

    void renderImg(int thickness = 1, cv::Scalar color = cv::Scalar(0, 0, 0));

private:
    float threshold_;
    int min_pixel_size_;

    float ring_threshold_ = -1.0f;
    int   ring_t_ = 1;

    cv::Mat* img_ = nullptr;

    Node root_;
    std::vector<Leaf> leaves_;

    IntegralSumsRGB sums_;
    int thread_count_ = 1;
    int split_depth_ = 3;

    std::vector<Leaf> frontier_;
    std::vector<uint8_t> frontier_is_leaf_;
    std::vector<std::vector<Leaf>> frontier_outputs_;
};

class QTreeWorker {
public:
    QTreeWorker();
    ~QTreeWorker();

    QTreeWorker(const QTreeWorker&) = delete;
    QTreeWorker& operator=(const QTreeWorker&) = delete;

    void submit(const cv::Mat& image,
                float threshold,
                int min_pixel_size,
                float ring_threshold,
                int ring_t,
                int thread_count,
                int split_depth,
                std::shared_ptr<void> source_owner = {});
    const std::vector<Leaf>& wait();

private:
    void run();

    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::thread worker_;
    bool stop_ = false;
    bool pending_ = false;
    bool complete_ = true;
    cv::Mat image_;
    float threshold_ = 0.f;
    int min_pixel_size_ = 1;
    float ring_threshold_ = -1.f;
    int ring_t_ = 1;
    int thread_count_ = 1;
    int split_depth_ = 3;
    std::shared_ptr<void> source_owner_;
    std::unique_ptr<QTree> qtree_;
    std::exception_ptr error_;
};

std::vector<Node> find_children(const Node& node);

void recursive_subdivide(Node& node, float threshold, int min_pixel_size, cv::Mat& img);

void build_integral_sums_rgb(const cv::Mat& img,
                             IntegralSumsRGB& out,
                             bool build_sumsq = true,
                             int thread_count = 1);

void build_leaf_id_lut(const std::vector<Leaf>& leaves,
                       int width,
                       int height,
                       std::vector<int32_t>& leaf_ids,
                       int thread_count = 1);

} 

#endif 
