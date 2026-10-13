/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file Frontend.hpp
 * @brief Header file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_FRONTEND_HPP_
#define INCLUDE_OKVIS_FRONTEND_HPP_

#include <mutex>

#include <okvis/Component.hpp>
#include <okvis/ViFrontendInterface.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include <thread>

namespace okvis {

class Frontend : public ViFrontendInterface {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Frontend(size_t numCameras, std::string dBowVocDir);
  virtual ~Frontend() override;

  bool loadComponent(std::string filename,
                     const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     bool componentFixed = true);

  virtual bool detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) override final;

  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const okvis::ViParameters & params,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) override final;

  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const override final;

  size_t getBriskDetectionOctaves() const {
    return briskDetectionOctaves_;
  }

  double getBriskDetectionThreshold() const {
    return briskDetectionThreshold_;
  }

  double getBriskDetectionAbsoluteThreshold() const {
    return briskDetectionAbsoluteThreshold_;
  }

  size_t getBriskDetectionMaximumKeypoints() const {
    return briskDetectionMaximumKeypoints_;
  }

  bool getBriskDescriptionRotationInvariance() const {
    return briskDescriptionRotationInvariance_;
  }

  bool getBriskDescriptionScaleInvariance() const {
    return briskDescriptionScaleInvariance_;
  }

  double getBriskMatchingThreshold() const {
    return briskMatchingThreshold_;
  }

  float getKeyframeInsertionOverlapThershold() const {
    return keyframeInsertionOverlapThreshold_;
  }

  bool isInitialized() {
    return isInitialized_;
  }

  void setBriskDetectionOctaves(size_t octaves) {
    briskDetectionOctaves_ = octaves;
    initialiseBriskFeatureDetectors();
  }

  void setBriskDetectionThreshold(double threshold) {
    briskDetectionThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  void setBriskDetectionAbsoluteThreshold(double threshold) {
    briskDetectionAbsoluteThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
    briskDetectionMaximumKeypoints_ = maxKeypoints;
    initialiseBriskFeatureDetectors();
  }

  void setBriskDescriptionRotationInvariance(bool invariance) {
    briskDescriptionRotationInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  void setBriskDescriptionScaleInvariance(bool invariance) {
    briskDescriptionScaleInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  void setBriskMatchingThreshold(double threshold) {
    briskMatchingThreshold_ = threshold;
  }

  void setKeyframeInsertionOverlapThreshold(float threshold) {
    keyframeInsertionOverlapThreshold_ = threshold;
  }

  void endCnnThreads();

  void clear();

private:

  std::vector<std::shared_ptr<cv::FeatureDetector> > featureDetectors_;

  std::vector<std::shared_ptr<cv::DescriptorExtractor> > descriptorExtractors_;

  std::vector<std::unique_ptr<std::mutex> > featureDetectorMutexes_;

  bool isInitialized_;        
  const size_t numCameras_;   

  size_t briskDetectionOctaves_;            
  double briskDetectionThreshold_;          
  double briskDetectionAbsoluteThreshold_;  
  size_t briskDetectionMaximumKeypoints_;   

  bool briskDescriptionRotationInvariance_; 
  bool briskDescriptionScaleInvariance_;    

  double briskMatchingThreshold_; 

  float keyframeInsertionOverlapThreshold_;  

  bool doWeNeedANewKeyframe(const Estimator& estimator,
                            std::shared_ptr<okvis::MultiFrame> currentFrame);

  template<class CAMERA_GEOMETRY>
  int matchToMap(Estimator& estimator,
                 const okvis::ViParameters& params,
                 const uint64_t currentFrameId,
                 const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively = nullptr);

  template<class CAMERA_GEOMETRY>
  void matchStereo(Estimator& estimator,
                   std::shared_ptr<okvis::MultiFrame> multiFrame,
                   const okvis::ViParameters& params,
                   bool asKeyframe);

  class DBoW;
  std::unique_ptr<DBoW> dBow_; 

  int getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                            const std::vector<std::vector<uchar>> &features,
                            std::vector<std::pair<StateId, double>> &stateIds) const;

  bool verifyRecognisedPlace(const Estimator &estimator,
                             const okvis::ViParameters &params,
                             const std::shared_ptr<const MultiFrame> framesInOut,
                             const std::shared_ptr<const MultiFrame> oldFrame,
                             kinematics::Transformation &T_Sold_Snew,
                             Eigen::Matrix<double, 6, 6>& H,
                             int minInliers = 10);

  bool runRansac3d2d(Estimator &estimator,
                    const okvis::cameras::NCameraSystem &nCameraSystem,
                    std::shared_ptr<okvis::MultiFrame> currentFrame,
                    bool initializePose,
                    bool removeOutliers);

  template <class CAMERA_GEOMETRY>
  int removeOutliers(Estimator &estimator,
                     const okvis::cameras::NCameraSystem &nCameraSystem,
                     std::shared_ptr<okvis::MultiFrame> currentFrame);

  int runRansac2d2d(Estimator& estimator,
                    const okvis::ViParameters& params, uint64_t currentFrameId,
                    uint64_t olderFrameId, bool initializePose,
                    bool removeOutliers, bool &rotationOnly);

  void initialiseBriskFeatureDetectors();

  template<class CAMERA_GEOMETRY>
  int matchMotionStereo(Estimator &estimator, const okvis::ViParameters &params,
                        const uint64_t currentFrameId, bool &rotationOnly);

  struct LandmarkToMatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p_W; 
    cv::Mat descriptors; 
    std::vector<KeypointIdentifier> kids; 
    Eigen::Matrix3Xd e_W; 
    Eigen::Matrix3Xd r_W; 
    bool is3d = false; 
    Eigen::Vector2d projection; 
    bool ignore = false; 
  };

  template<class CAMERA_GEOMETRY>
  void matchToMapByThread(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs,
      std::vector<double>& reprErrs) const;

  template<class CAMERA_GEOMETRY>
  void matchToMapByThreadUnitialised(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs) const;

  std::atomic_bool trackingLost_; 

  std::map<StateId,std::vector<std::thread*>> cnnThreads_;

  AlignedVector<Component> components_; 
  std::vector<std::unique_ptr<DBoW>> componentDBows_; 
  std::vector<bool> componentsFixed_; 
};

}  

#endif 
