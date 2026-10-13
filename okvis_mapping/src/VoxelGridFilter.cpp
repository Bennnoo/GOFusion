

#include <okvis/VoxelGridFilter.hpp>

namespace okvis{

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VoxelHashMap::Pointcloud() const {
      Vector3dVector points;
      points.reserve(max_points_per_voxel_ * map_.size());
      for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
          points.push_back(point);
        }
      }
      return points;
    }

    void VoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin) {
      AddPoints(points);
      RemovePointsFarFromLocation(origin);
    }

    void VoxelHashMap::Update(const Vector3dVector &points, const okvis::kinematics::Transformation &pose) {
      Vector3dVector points_transformed(points.size());
      std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                     [&](const auto &point) { return pose.T3x4() * point.homogeneous(); });
      const Eigen::Vector3d &origin = pose.r();
      Update(points_transformed, origin);
    }

    void VoxelHashMap::AddPoints(const Vector3dVector &points) {
      std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
          auto voxel = Voxel((point / voxel_size_).template cast<int>());
          auto search = map_.find(voxel);
          if (search != map_.end()) {
            auto &voxel_block = search->second;
            voxel_block.AddPoint(point);
          } else {
            map_.insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
          }
      });
    }

    void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
      for (const auto &[voxel, voxel_block] : map_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
          map_.erase(voxel);
        }
      }
    }

}