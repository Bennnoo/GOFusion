/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
 */

#include "OkvisOccupancyViewer3D.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#if __has_include(<vtkObject.h>)
#include <vtkObject.h>
#define OKVIS_HAS_VTKOBJECT 1
#else
#define OKVIS_HAS_VTKOBJECT 0
#endif
#if __has_include(<vtkMatrix4x4.h>) && __has_include(<vtkLODActor.h>)
#include <vtkMatrix4x4.h>
#include <vtkLODActor.h>
#define OKVIS_HAS_VTK_CLOUD_POSE 1
#else
#define OKVIS_HAS_VTK_CLOUD_POSE 0
#endif
#if __has_include(<vtkCommand.h>) && __has_include(<vtkCoordinate.h>)     && __has_include(<vtkRenderWindow.h>) && __has_include(<vtkRenderWindowInteractor.h>)     && __has_include(<vtkSliderRepresentation2D.h>) && __has_include(<vtkSliderWidget.h>)     && __has_include(<vtkSmartPointer.h>)
#include <vtkCommand.h>
#include <vtkCoordinate.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkSmartPointer.h>
#define OKVIS_HAS_VTK_SLIDERS 1
#else
#define OKVIS_HAS_VTK_SLIDERS 0
#endif

namespace {
const std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>& submapPalette()
{
  static const std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> palette = {
      Eigen::Vector3i(255, 30, 30),   
      Eigen::Vector3i(30, 220, 30),   
      Eigen::Vector3i(35, 95, 255),   
      Eigen::Vector3i(255, 210, 30),  
      Eigen::Vector3i(230, 45, 220),  
      Eigen::Vector3i(20, 230, 230),  
      Eigen::Vector3i(255, 120, 20),  
      Eigen::Vector3i(145, 65, 255),  
      Eigen::Vector3i(145, 255, 35),  
      Eigen::Vector3i(255, 65, 145),  
      Eigen::Vector3i(30, 170, 255),  
      Eigen::Vector3i(170, 110, 30)   
  };
  return palette;
}

std::string occupiedCloudId(const uint64_t submap_id)
{
  return "occupied_submap_" + std::to_string(submap_id);
}

constexpr const char* kCurrentPoseCoordinateId = "current_pose";
constexpr const char* kCurrentPoseAxisXId = "current_pose_axis_x";
constexpr const char* kCurrentPoseAxisYId = "current_pose_axis_y";
constexpr const char* kCurrentPoseAxisZId = "current_pose_axis_z";
constexpr const char* kCurrentPoseOriginId = "current_pose_origin";

bool setCloudPose(const std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
                  const std::string& cloud_id,
                  const Eigen::Matrix4f& T_WK)
{
#if OKVIS_HAS_VTK_CLOUD_POSE
  auto actor_map = viewer->getCloudActorMap();
  if (!actor_map) {
    return false;
  }
  auto it = actor_map->find(cloud_id);
  if (it == actor_map->end() || !it->second.actor) {
    return false;
  }
  vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matrix->SetElement(r, c, static_cast<double>(T_WK(r, c)));
    }
  }
  it->second.actor->SetUserMatrix(matrix);
  it->second.actor->Modified();
  return true;
#else
  (void) viewer;
  (void) cloud_id;
  (void) T_WK;
  return false;
#endif
}

#if OKVIS_HAS_VTK_SLIDERS
class SliderCallback : public vtkCommand {
 public:
  static SliderCallback* New() { return new SliderCallback(); }
  std::function<void(double)> callback;

  void Execute(vtkObject* caller, unsigned long, void*) override
  {
    auto* slider = vtkSliderWidget::SafeDownCast(caller);
    if (!slider || !callback) {
      return;
    }
    auto* rep = vtkSliderRepresentation::SafeDownCast(slider->GetRepresentation());
    if (!rep) {
      return;
    }
    callback(rep->GetValue());
  }
};

vtkSmartPointer<vtkSliderWidget> createSliderWidget(
    vtkRenderWindowInteractor* interactor, const std::string& title,
    const double min_value, const double max_value, const double value,
    const double x0, const double y0, const double x1, const double y1,
    const std::function<void(double)>& callback,
    std::vector<vtkSmartPointer<vtkCommand>>& callbacks_keeper)
{
  auto rep = vtkSmartPointer<vtkSliderRepresentation2D>::New();
  rep->SetMinimumValue(min_value);
  rep->SetMaximumValue(max_value);
  rep->SetValue(value);
  rep->SetTitleText(title.c_str());
  rep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
  rep->GetPoint1Coordinate()->SetValue(x0, y0);
  rep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
  rep->GetPoint2Coordinate()->SetValue(x1, y1);
  rep->SetSliderLength(0.015);
  rep->SetSliderWidth(0.02);
  rep->SetEndCapLength(0.01);
  rep->SetEndCapWidth(0.02);
  rep->SetTubeWidth(0.006);
  rep->SetLabelFormat("%.0f");
  rep->SetTitleHeight(0.016);
  rep->SetLabelHeight(0.012);

  auto slider_widget = vtkSmartPointer<vtkSliderWidget>::New();
  slider_widget->SetInteractor(interactor);
  slider_widget->SetRepresentation(rep);
  if (min_value == 0.0 && max_value == 1.0) {
    slider_widget->SetAnimationModeToJump();
  } else {
    slider_widget->SetAnimationModeToAnimate();
  }
  slider_widget->EnabledOn();

  auto cb = vtkSmartPointer<SliderCallback>::New();
  cb->callback = callback;
  callbacks_keeper.push_back(cb);
  slider_widget->AddObserver(vtkCommand::InteractionEvent, cb);
  return slider_widget;
}
#endif

} 

namespace okvis {

OkvisOccupancyViewer3D::OkvisOccupancyViewer3D(const Options& options)
    : options_(options)
{
  options_.max_fps = std::max(1, options_.max_fps);
  options_.max_points = std::max(1000, options_.max_points);
  options_.occupied_point_size = std::max(1, options_.occupied_point_size);
  options_.fisher_point_size = std::max(1, options_.fisher_point_size);
  options_.window_width = std::max(1, options_.window_width);
  options_.window_height = std::max(1, options_.window_height);

  occupied_point_size_.store(options_.occupied_point_size);

  fisher_point_size_.store(options_.fisher_point_size);
  use_fisher_overlay_.store(options_.use_fisher_overlay && !options_.use_global_fisher_overlay);
  use_global_fisher_overlay_.store(options_.use_global_fisher_overlay);

  if (options_.window_name.empty()) {
    options_.window_name = "GO Occupancy";
  }
}

OkvisOccupancyViewer3D::~OkvisOccupancyViewer3D()
{
  stop();
}

void OkvisOccupancyViewer3D::start()
{
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  user_stop_requested_.store(false);
  worker_ = std::thread(&OkvisOccupancyViewer3D::runLoop, this);
}

void OkvisOccupancyViewer3D::stop()
{
  running_.store(false);
  if (worker_.joinable()) {
    worker_.join();
  }
}

void OkvisOccupancyViewer3D::discardScene()
{
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = OccupancyPreviewFrame();
    has_frame_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    latest_trajectory_.clear();
    has_trajectory_ = false;
  }

  discard_scene_requested_.store(true, std::memory_order_release);
}

bool OkvisOccupancyViewer3D::running() const
{
  return running_.load();
}

void OkvisOccupancyViewer3D::submitFrame(const OccupancyPreviewFrame& frame)
{
  if (!running_.load()) {
    return;
  }
  std::unique_lock<std::mutex> lock(frame_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {

    return;
  }
  latest_frame_ = frame;
  has_frame_ = true;
}

void OkvisOccupancyViewer3D::submitTrajectory(
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& trajectory)
{
  if (!running_.load()) {
    return;
  }
  std::unique_lock<std::mutex> lock(trajectory_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    return;
  }
  latest_trajectory_ = trajectory;
  has_trajectory_ = true;
}

void OkvisOccupancyViewer3D::runLoop()
{
#if OKVIS_HAS_VTKOBJECT
  vtkObject::GlobalWarningDisplayOff();
#endif
  auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>(options_.window_name);
  viewer->setBackgroundColor(0.04, 0.04, 0.04);
  viewer->setSize(options_.window_width, options_.window_height);
  if (viewer->getRenderWindow()) {
    viewer->getRenderWindow()->SetSize(options_.window_width, options_.window_height);
  }
  viewer->initCameraParameters();
  viewer->setCameraClipDistances(0.01, 2000.0);
  viewer->registerKeyboardCallback(
      [this](const pcl::visualization::KeyboardEvent& event) {
        if (!event.keyDown()) return;
        const std::string& symbol = event.getKeySym();
        if (event.getKeyCode() == 27 || symbol == "Escape"
            || symbol == "q" || symbol == "Q") {
          user_stop_requested_.store(true);
        }
      });

#if OKVIS_HAS_VTK_SLIDERS
  std::vector<vtkSmartPointer<vtkSliderWidget>> slider_widgets;
  std::vector<vtkSmartPointer<vtkCommand>> slider_callbacks;
  vtkSmartPointer<vtkSliderWidget> fisher_slider;
  vtkSmartPointer<vtkSliderWidget> global_fisher_slider;
  std::atomic_bool use_raw_occupied_color(true);
  if (viewer->getRenderWindow() && viewer->getRenderWindow()->GetInteractor()) {
    auto* interactor = viewer->getRenderWindow()->GetInteractor();
    slider_widgets.push_back(createSliderWidget(
        interactor, "Occupied Size", 1.0, 30.0,
        static_cast<double>(occupied_point_size_.load()),
        0.03, 0.44, 0.25, 0.44,
        [this](const double v) {
          occupied_point_size_.store(std::max(1, static_cast<int>(std::lround(v))));
        }, slider_callbacks));
    slider_widgets.push_back(createSliderWidget(
        interactor, "Use Raw Color", 0.0, 1.0, 1.0,
        0.03, 0.37, 0.25, 0.37,
        [&use_raw_occupied_color](const double v) {
          use_raw_occupied_color.store(v >= 0.5);
        }, slider_callbacks));

    fisher_slider = createSliderWidget(
        interactor, "Fisher uncertainty", 0.0, 1.0, use_fisher_overlay_.load() ? 1.0 : 0.0,
        0.03, 0.30, 0.25, 0.30,
        [this, &global_fisher_slider](const double v) {
          if (v >= 0.5) {
            use_global_fisher_overlay_.store(false);
            auto* rep = vtkSliderRepresentation::SafeDownCast(
                global_fisher_slider->GetRepresentation());
            rep->SetValue(0.0);
          }
          use_fisher_overlay_.store(v >= 0.5);
        }, slider_callbacks);
    slider_widgets.push_back(fisher_slider);

    global_fisher_slider = createSliderWidget(
        interactor, "Global Fisher uncertainty", 0.0, 1.0,
        use_global_fisher_overlay_.load() ? 1.0 : 0.0,
        0.03, 0.23, 0.25, 0.23,
        [this, &fisher_slider](const double v) {
          if (v >= 0.5) {
            use_fisher_overlay_.store(false);
            auto* rep = vtkSliderRepresentation::SafeDownCast(
                fisher_slider->GetRepresentation());
            rep->SetValue(0.0);
          }
          use_global_fisher_overlay_.store(v >= 0.5);
        }, slider_callbacks);
    slider_widgets.push_back(global_fisher_slider);
    slider_widgets.push_back(createSliderWidget(
        interactor, "Fisher Size", 1.0, 30.0,
        static_cast<double>(fisher_point_size_.load()),
        0.03, 0.16, 0.25, 0.16,
        [this](const double v) {
          fisher_point_size_.store(std::max(1, static_cast<int>(std::lround(v))));
        }, slider_callbacks));
  }
#endif

  const auto min_frame_dt =
      std::chrono::milliseconds(static_cast<int>(std::max(1, 1000 / options_.max_fps)));
  bool initial_view_set = false;
  bool enforce_initial_window_size = true;
  int initial_size_guard_frames = 90;
  int initial_size_stable_frames = 0;
  std::unordered_map<uint64_t, size_t> submap_color_indices;
  size_t next_submap_color_index = 0;
  std::unordered_map<uint64_t, uint64_t> rendered_geometry_revision;
  std::unordered_map<uint64_t, uint64_t> rendered_colour_revision;
  std::unordered_set<uint64_t> visible_submaps;
  OccupancyPreviewFrame cached_frame;
  bool has_cached_frame = false;
  bool last_use_raw_occupied_color = use_raw_occupied_color.load();
  bool last_use_fisher_overlay = use_fisher_overlay_.load();
  bool last_use_global_fisher_overlay = use_global_fisher_overlay_.load();

  int last_occupied_point_size = std::max(1, occupied_point_size_.load());

  int last_fisher_point_size = std::max(1, fisher_point_size_.load());

  while (running_.load()) {
    const auto tic = std::chrono::steady_clock::now();

    if (discard_scene_requested_.exchange(false, std::memory_order_acq_rel)) {
      for (const auto& entry : rendered_geometry_revision) {
        viewer->removePointCloud(occupiedCloudId(entry.first));
      }

      viewer->removePointCloud("trajectory_cloud");
      viewer->removeCoordinateSystem(kCurrentPoseCoordinateId);
      viewer->removeShape(kCurrentPoseAxisXId);
      viewer->removeShape(kCurrentPoseAxisYId);
      viewer->removeShape(kCurrentPoseAxisZId);
      viewer->removeShape(kCurrentPoseOriginId);
      rendered_geometry_revision.clear();
      rendered_colour_revision.clear();
      submap_color_indices.clear();
      visible_submaps.clear();
      cached_frame = OccupancyPreviewFrame();
      has_cached_frame = false;
      initial_view_set = false;
    }

    OccupancyPreviewFrame frame;
    bool has_new_frame = false;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (has_frame_) {
        frame = latest_frame_;
        has_new_frame = true;
        has_frame_ = false;
      }
    }

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> trajectory;
    bool has_new_trajectory = false;
    {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      if (has_trajectory_) {
        trajectory = latest_trajectory_;
        has_new_trajectory = true;
        has_trajectory_ = false;
      }
    }


    const int occupied_point_size = std::max(1, occupied_point_size_.load());

    const int fisher_point_size = std::max(1, fisher_point_size_.load());
    const bool use_raw_occupied = use_raw_occupied_color.load();
    const bool use_fisher_overlay = use_fisher_overlay_.load();
    const bool use_global_fisher_overlay = use_global_fisher_overlay_.load();
    const bool use_any_fisher_overlay = use_fisher_overlay || use_global_fisher_overlay;

    bool controls_changed =
        (occupied_point_size != last_occupied_point_size)
        || (fisher_point_size != last_fisher_point_size);
    controls_changed = controls_changed || (use_raw_occupied != last_use_raw_occupied_color);
    controls_changed = controls_changed || (use_fisher_overlay != last_use_fisher_overlay);
    controls_changed =
        controls_changed || (use_global_fisher_overlay != last_use_global_fisher_overlay);
    if (has_new_frame) {
      cached_frame = frame;
      has_cached_frame = true;
    }

    if (has_cached_frame && (has_new_frame || controls_changed)) {
      const auto& render_frame = cached_frame;
      visible_submaps.clear();
      const auto& palette = submapPalette();
      if (render_frame.submap_clouds) {
        for (const auto& submap_cloud : *render_frame.submap_clouds) {
          const uint64_t submap_id = submap_cloud.submap_id;
          visible_submaps.insert(submap_id);
          auto color_it = submap_color_indices.find(submap_id);
          if (color_it == submap_color_indices.end()) {
            color_it = submap_color_indices.emplace(submap_id, next_submap_color_index++).first;
          }
          const Eigen::Vector3i palette_rgb = palette[color_it->second % palette.size()];
          const std::string occ_id = occupiedCloudId(submap_id);

          const uint64_t last_revision = rendered_geometry_revision.count(submap_id)
                                             ? rendered_geometry_revision[submap_id]
                                             : 0;
          const bool geometry_changed =
              (submap_cloud.geometry_revision != last_revision) || submap_cloud.geometry_updated;
          const uint64_t last_colour_revision = rendered_colour_revision.count(submap_id)
                                                    ? rendered_colour_revision[submap_id]
                                                    : 0;
          const bool colour_changed = submap_cloud.colour_revision != last_colour_revision;
          const bool refresh_occupied = geometry_changed || colour_changed || controls_changed;
          if (refresh_occupied) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_cloud(
                new pcl::PointCloud<pcl::PointXYZRGB>());
            if (submap_cloud.occupied_points_local) {
              const auto& points = *submap_cloud.occupied_points_local;
              const bool has_raw_colours =
                  submap_cloud.occupied_colours
                  && submap_cloud.occupied_colours->size() == points.size();
              const bool has_fisher_colours =
                  submap_cloud.occupied_fisher_colours
                  && submap_cloud.occupied_fisher_colours->size() == points.size();
              const bool has_global_fisher_colours =
                  submap_cloud.occupied_global_fisher_colours
                  && submap_cloud.occupied_global_fisher_colours->size() == points.size();
              const auto* fisher_colours =
                  has_fisher_colours ? submap_cloud.occupied_fisher_colours.get() : nullptr;
              const auto* global_fisher_colours =
                  has_global_fisher_colours ? submap_cloud.occupied_global_fisher_colours.get()
                                            : nullptr;
              const auto* active_fisher_colours =
                  use_global_fisher_overlay ? global_fisher_colours
                                           : (use_fisher_overlay ? fisher_colours : nullptr);
              const auto* raw_colours = has_raw_colours
                                            ? submap_cloud.occupied_colours.get()
                                            : nullptr;
              local_cloud->reserve(points.size());
              for (size_t i = 0; i < points.size(); ++i) {
                const Eigen::Vector3f& p = points[i];
                pcl::PointXYZRGB point;
                point.x = p.x();
                point.y = p.y();
                point.z = p.z();
                Eigen::Vector3i rgb = palette_rgb;
                if (use_any_fisher_overlay && active_fisher_colours) {
                  rgb = (*active_fisher_colours)[i];
                  if (use_raw_occupied && raw_colours) {
                    const Eigen::Vector3i& raw_rgb = (*raw_colours)[i];
                    rgb = (((raw_rgb.cast<float>() * 0.45f)
                            + (rgb.cast<float>() * 0.55f))
                               .array()
                               .round())
                              .matrix()
                              .cast<int>();
                  }
                } else if (use_any_fisher_overlay) {
                  if (use_raw_occupied && raw_colours) {
                    rgb = (*raw_colours)[i];
                  } else {
                    rgb = Eigen::Vector3i(70, 70, 70);
                  }
                } else if (use_raw_occupied && raw_colours) {
                  rgb = (*raw_colours)[i];
                }
                point.r = static_cast<uint8_t>(rgb.x());
                point.g = static_cast<uint8_t>(rgb.y());
                point.b = static_cast<uint8_t>(rgb.z());
                local_cloud->push_back(point);
              }
            }

            if (!viewer->updatePointCloud(local_cloud, occ_id)) {
              viewer->addPointCloud(local_cloud, occ_id);
            }
          }
          const int active_occupied_point_size =
              use_any_fisher_overlay ? fisher_point_size : occupied_point_size;
          viewer->setPointCloudRenderingProperties(
              pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
              static_cast<double>(active_occupied_point_size), occ_id);
          setCloudPose(viewer, occ_id, submap_cloud.T_WK);

          rendered_geometry_revision[submap_id] = submap_cloud.geometry_revision;
          rendered_colour_revision[submap_id] = submap_cloud.colour_revision;
        }
      }

      for (auto it = rendered_geometry_revision.begin(); it != rendered_geometry_revision.end();) {
        if (visible_submaps.count(it->first) == 0) {
          viewer->removePointCloud(occupiedCloudId(it->first));
          submap_color_indices.erase(it->first);
          rendered_colour_revision.erase(it->first);
          it = rendered_geometry_revision.erase(it);
        } else {
          ++it;
        }
      }

      viewer->removeShape("occupancy_stats");
      std::ostringstream ss;
      ss << "submaps=" << render_frame.num_submaps
         << " occupied_points=" << render_frame.num_points
         << " integration=" << render_frame.integration_counter;
      viewer->addText(ss.str(), 10, 10, 14, 1.0, 1.0, 1.0, "occupancy_stats");
      viewer->removeShape("occupancy_controls_hint");
      std::ostringstream controls_ss;
      controls_ss << "controls: occ_size=" << occupied_point_size
                  << " fisher_size=" << fisher_point_size
                  << " raw=" << (use_raw_occupied ? "on" : "off")
                  << " fisher_local=" << (use_fisher_overlay ? "on" : "off")
                  << " fisher_global=" << (use_global_fisher_overlay ? "on" : "off")
                  << " fisher_mode="
                  << (use_global_fisher_overlay ? "global"
                                                : (use_fisher_overlay ? "local" : "off"));
      viewer->addText(controls_ss.str(), 10, 30, 12, 0.9, 0.9, 0.9, "occupancy_controls_hint");

      viewer->removeCoordinateSystem(kCurrentPoseCoordinateId);
      viewer->removeShape(kCurrentPoseAxisXId);
      viewer->removeShape(kCurrentPoseAxisYId);
      viewer->removeShape(kCurrentPoseAxisZId);
      viewer->removeShape(kCurrentPoseOriginId);
      if (options_.show_pose) {
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        pose.matrix() = render_frame.T_WS;
        Eigen::Affine3f pose_display = pose;

        pose_display.linear() *=
            Eigen::AngleAxisf(static_cast<float>(M_PI), Eigen::Vector3f::UnitZ()).toRotationMatrix();

        const Eigen::Vector3f origin = pose_display.translation();
        const Eigen::Matrix3f R_display = pose_display.linear();
        constexpr float kPoseAxisLength = 0.32f;
        constexpr float kPoseAxisRadius = 0.02f;
        constexpr double kPoseAxisWidth = 10.0;
        const Eigen::Vector3f px = origin + kPoseAxisLength * R_display.col(0);
        const Eigen::Vector3f py = origin + kPoseAxisLength * R_display.col(1);
        const Eigen::Vector3f pz = origin + kPoseAxisLength * R_display.col(2);

        viewer->addLine(pcl::PointXYZ(origin.x(), origin.y(), origin.z()),
                        pcl::PointXYZ(px.x(), px.y(), px.z()),
                        1.0, 0.0, 0.0, kCurrentPoseAxisXId);
        viewer->addLine(pcl::PointXYZ(origin.x(), origin.y(), origin.z()),
                        pcl::PointXYZ(py.x(), py.y(), py.z()),
                        0.0, 1.0, 0.0, kCurrentPoseAxisYId);
        viewer->addLine(pcl::PointXYZ(origin.x(), origin.y(), origin.z()),
                        pcl::PointXYZ(pz.x(), pz.y(), pz.z()),
                        0.0, 0.0, 1.0, kCurrentPoseAxisZId);
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, kPoseAxisWidth, kCurrentPoseAxisXId);
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, kPoseAxisWidth, kCurrentPoseAxisYId);
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, kPoseAxisWidth, kCurrentPoseAxisZId);
        viewer->addSphere(pcl::PointXYZ(origin.x(), origin.y(), origin.z()),
                          kPoseAxisRadius, 1.0, 1.0, 1.0, kCurrentPoseOriginId);
      }

      if (!initial_view_set) {
        const Eigen::Vector3f t = render_frame.T_WS.block<3, 1>(0, 3);
        const Eigen::Matrix3f R = render_frame.T_WS.block<3, 3>(0, 0);
        const Eigen::Vector3f forward = R.col(2).normalized();
        const Eigen::Vector3f up = (-R.col(1)).normalized();
        const Eigen::Vector3f focal = t + 2.0f * forward;
        viewer->setCameraPosition(t.x(), t.y(), t.z(),
                                  focal.x(), focal.y(), focal.z(),
                                  up.x(), up.y(), up.z());
        initial_view_set = true;
      }


      last_occupied_point_size = occupied_point_size;

      last_fisher_point_size = fisher_point_size;
      last_use_raw_occupied_color = use_raw_occupied;
      last_use_fisher_overlay = use_fisher_overlay;
      last_use_global_fisher_overlay = use_global_fisher_overlay;
    }

    if (has_new_trajectory) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr traj_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>());
      traj_cloud->reserve(trajectory.size());
      const size_t N = trajectory.size();
      for (size_t i = 0; i < N; ++i) {
        const Eigen::Vector3f& p = trajectory[i];
        const float t = (N > 1) ? static_cast<float>(i) / static_cast<float>(N - 1) : 1.0f;
        pcl::PointXYZRGB point;
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();
        point.r = static_cast<uint8_t>(std::round(80.0f * (1.0f - t) + 255.0f * t));
        point.g = static_cast<uint8_t>(std::round(220.0f * (1.0f - t) + 70.0f * t));
        point.b = static_cast<uint8_t>(std::round(255.0f * (1.0f - t) + 40.0f * t));
        traj_cloud->push_back(point);
      }

      if (!viewer->updatePointCloud(traj_cloud, "trajectory_cloud")) {
        viewer->addPointCloud(traj_cloud, "trajectory_cloud");
      }
      viewer->setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4.0, "trajectory_cloud");
    }

    viewer->spinOnce(1, false);
    if (enforce_initial_window_size && viewer->getRenderWindow()) {

      int* sz = viewer->getRenderWindow()->GetSize();
      if (!sz || sz[0] != options_.window_width || sz[1] != options_.window_height) {
        viewer->setSize(options_.window_width, options_.window_height);
        viewer->getRenderWindow()->SetSize(options_.window_width, options_.window_height);
        initial_size_stable_frames = 0;
      } else {
        ++initial_size_stable_frames;
      }
      --initial_size_guard_frames;
      if (initial_size_stable_frames >= 15 || initial_size_guard_frames <= 0) {
        enforce_initial_window_size = false;
      }
    }
    if (viewer->wasStopped()) {
      running_.store(false);
      break;
    }

    const auto elapsed = std::chrono::steady_clock::now() - tic;
    if (elapsed < min_frame_dt) {
      std::this_thread::sleep_for(min_frame_dt - elapsed);
    }
  }
}

}  
