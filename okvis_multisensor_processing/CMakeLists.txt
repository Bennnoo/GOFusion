# SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.8...3.30)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc highgui calib3d)
find_package(PCL REQUIRED)
find_package(OpenMP QUIET)

# Generate the configured header file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/okvisConfig.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/include/okvis/okvisConfig.hpp" @ONLY)

set(FRAME_SLOT_LIB_NAME okvis_shared_frame_slot)
add_library(${FRAME_SLOT_LIB_NAME}
  include/okvis/SharedFrameSlot.hpp
  src/SharedFrameSlot.cpp)
target_include_directories(${FRAME_SLOT_LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${GLOG_INCLUDE_DIRS})
target_link_libraries(${FRAME_SLOT_LIB_NAME}
  PUBLIC okvis::Common ${OpenCV_LIBS}
  PRIVATE ${GLOG_LIBRARIES})
if(NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${FRAME_SLOT_LIB_NAME} PUBLIC CUDA::cudart)
endif()
target_compile_features(${FRAME_SLOT_LIB_NAME}
  PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${FRAME_SLOT_LIB_NAME}
  PUBLIC ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE ${OKVIS_PRIVATE_CXX_FLAGS})
add_library(okvis::SharedFrameSlot ALIAS ${FRAME_SLOT_LIB_NAME})

set(LIB_NAME okvis_multisensor_processing)
add_library(${LIB_NAME}
  src/XDatasetReader.cpp
  src/SubmappingInterface.cpp
  src/ThreadedSlam.cpp
  src/TrackingImageDownsampler.cpp
  src/TrajectoryOutput.cpp
  src/ViVisualizer.cpp
  src/GsFisherSnapshot.cpp
  src/GsGeometryTargetCache.cpp
)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OKVIS_RGBD_ZSTD REQUIRED IMPORTED_TARGET libzstd)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GLOG_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
  Boost::filesystem
  Eigen3::Eigen
  okvis::CV
  okvis::Ceres
  okvis::Common
  okvis::Frontend
  okvis::Mapping
  okvis::Kinematics
  okvis::Time
  okvis::Timing
  okvis::Util
  okvis::SharedFrameSlot
  ${PCL_LIBRARIES}
  PkgConfig::OKVIS_RGBD_ZSTD
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${LIB_NAME} OpenMP::OpenMP_CXX)
endif()
if(NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${LIB_NAME} CUDA::cudart)
endif()
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)

# Avoid a GCC 13 optimizer failure in this translation unit.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  set_property(SOURCE src/SubmappingInterface.cpp APPEND PROPERTY
    COMPILE_OPTIONS -fno-tree-vrp -fno-ipa-vrp)
endif()

add_library(okvis::MultisensorProcessing ALIAS ${LIB_NAME})

install(TARGETS ${FRAME_SLOT_LIB_NAME} ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
