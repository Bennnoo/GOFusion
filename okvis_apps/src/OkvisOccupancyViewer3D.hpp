/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 */

#ifndef OKVIS_APPS_OKVIS_OCCUPANCY_VIEWER_3D_HPP
#define OKVIS_APPS_OKVIS_OCCUPANCY_VIEWER_3D_HPP

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include <okvis/SubmappingInterface.hpp>

namespace okvis {

class OkvisOccupancyViewer3D {
 public:
  struct Options {
    std::string window_name = "GO Occupancy";
    int max_fps = 15;
    bool show_pose = true;

    int occupied_point_size = 2;

    int fisher_point_size = 2;
    int max_points = 400000;
    int window_width = 960;
    int window_height = 600;
    bool use_fisher_overlay = false;
    bool use_global_fisher_overlay = false;

  };

  explicit OkvisOccupancyViewer3D(const Options& options);
  ~OkvisOccupancyViewer3D();

  OkvisOccupancyViewer3D(const OkvisOccupancyViewer3D&) = delete;
  OkvisOccupancyViewer3D& operator=(const OkvisOccupancyViewer3D&) = delete;

  void start();
  void stop();

  void discardScene();
  bool running() const;
  bool userStopRequested() const { return user_stop_requested_.load(); }
  bool useFisherOverlayEnabled() const { return use_fisher_overlay_.load(); }
  bool useGlobalFisherOverlayEnabled() const
  {
    return use_global_fisher_overlay_.load();
  }

  void submitFrame(const OccupancyPreviewFrame& frame);
  void submitTrajectory(
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& trajectory);

 private:
  void runLoop();

  Options options_;
  std::atomic_bool running_{false};
  std::atomic_bool user_stop_requested_{false};
  std::atomic_bool discard_scene_requested_{false};
  std::thread worker_;

  std::mutex frame_mutex_;
  OccupancyPreviewFrame latest_frame_;
  bool has_frame_ = false;

  std::mutex trajectory_mutex_;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> latest_trajectory_;
  bool has_trajectory_ = false;


  std::atomic_int occupied_point_size_{2};

  std::atomic_int fisher_point_size_{2};
  std::atomic_bool use_fisher_overlay_{false};
  std::atomic_bool use_global_fisher_overlay_{false};

};

}  

#endif  
