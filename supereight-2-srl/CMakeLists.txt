# SPDX-FileCopyrightText: 2021-2023 Smart Robotics Lab, Imperial College London, Technical University of Munich
# SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)

cmake_minimum_required(VERSION 3.10...3.16)
project(Supereight2 VERSION 0.27.1 LANGUAGES CXX)

option(SE_OPENMP "Compile supereight with OpenMP" ON)
option(SE_ASSERTS "Compile supereight with runtime assertions enabled" OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Boost REQUIRED) # We only need Boost.Pool which is a header-only library.
find_package(TBB COMPONENTS tbb)
if(TBB_tbb_FOUND)
    set(SE_TBB 1)
endif()
find_library(MATH_LIB m)

# Compiler options
set(SUPPORTED_COMPILER "$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>")
set(SE_PRIVATE_CXX_FLAGS "$<${SUPPORTED_COMPILER}:-Wall;-Wextra;-Wpedantic;-Wno-unknown-pragmas>")

# Compile the dependencies
add_subdirectory(third_party)

# Compile the supereight library
set(LIB_NAME "supereight2")
add_library(${LIB_NAME} STATIC
    "src/common/colour_utils.cpp"
    "src/common/system_utils.cpp"
    "src/common/image_utils.cpp"
    "src/common/perfstats.cpp"
    "src/common/str_utils.cpp"
    "src/common/yaml.cpp"
    "src/image/image.cpp"
    "src/map/data.cpp"
    "src/map/io/mesh_io.cpp"
    "src/map/preprocessor.cpp"
    "src/map/raycaster.cpp"
    "src/sensor/lidar.cpp"
    "src/sensor/range_image_lidar.cpp"
    "src/sensor/pinhole_camera.cpp"
)
configure_file("include/se/supereight_config.hpp.in" "include/se/supereight_config.hpp" @ONLY)
# Set the C++ standard required by the library.
target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
target_compile_options(${LIB_NAME} PRIVATE ${SE_PRIVATE_CXX_FLAGS})
target_include_directories(${LIB_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
    PUBLIC
      Eigen3::Eigen
      Boost::boost
      ${OpenCV_LIBS}
      ${TBB_IMPORTED_TARGETS}
      SRL::Projection
)
if(OKVIS_ENABLE_GS_MAPPING)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(se2_tinyply STATIC third_party/tinyply/source/tinyply.cpp)
    target_include_directories(se2_tinyply
        PUBLIC
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyply/source
    )

    add_library(se2_json INTERFACE)
    target_include_directories(se2_json
        INTERFACE
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
    )

    add_library(se2_glm INTERFACE)
    target_include_directories(se2_glm
        INTERFACE
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    )

    set(GS_LIB_NAME "gsmodel")
    add_library(${GS_LIB_NAME} STATIC
        src/gs/quad_tree.cpp
        src/gs/error_driven_densify.cpp
        src/gs/gaussian.cu
        src/gs/fastergs_rasterizer.cu
        src/gs/monogs_rasterizer.cu
        src/gs/monogs_rasterize_points.cu
        src/gs/preview_converter.cu
        src/gs/rgb_ingest.cu
        fastergs_cuda_rasterizer/src/backward.cu
        fastergs_cuda_rasterizer/src/forward.cu
        fastergs_cuda_rasterizer/src/rasterization_api.cu
        mono_cuda_rasterizer/src/backward.cu
        mono_cuda_rasterizer/src/forward.cu
        mono_cuda_rasterizer/src/rasterizer_impl.cu
    )
    target_include_directories(${GS_LIB_NAME}
        PUBLIC
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyply/source
        PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/mono_cuda_rasterizer/include
          ${CMAKE_CURRENT_SOURCE_DIR}/fastergs_cuda_rasterizer/include
    )
    target_link_libraries(${GS_LIB_NAME}
        PUBLIC
          ${TORCH_LIBRARIES}
          CUDA::cudart
          CUDA::curand
          CUDA::cublas
          se2_tinyply
          se2_json
          se2_glm
          Eigen3::Eigen
          ${OpenCV_LIBS}
    )
    find_library(SE2_NVML_LIBRARY nvidia-ml)
    if(SE2_NVML_LIBRARY)
        target_link_libraries(${GS_LIB_NAME} PUBLIC ${SE2_NVML_LIBRARY})
    endif()

    # Keep install/export of Supereight2 unchanged: expose GS only in the build tree.
    target_link_libraries(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${GS_LIB_NAME}>)
    target_compile_definitions(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:OKVIS_ENABLE_GS_MAPPING=1>)

endif()
if(MATH_LIB)
    target_link_libraries(${LIB_NAME} PUBLIC ${MATH_LIB})
endif()
# Find OpenMP, warn if disabled.
if(SE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        target_link_libraries(${LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
        if(TARGET gsmodel)
            target_link_libraries(gsmodel PRIVATE OpenMP::OpenMP_CXX)
        endif()
        message(STATUS "Compiling with OpenMP support")
    else()
        message(WARNING "OpenMP not found. Performance may be terrible.")
    endif()
else()
    message(WARNING "Building without OpenMP. Performance may be terrible.")
endif()
if(SE_ASSERTS)
    target_compile_options(${LIB_NAME} PUBLIC -UNDEBUG)
endif()
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") AND ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 9.0))
    target_link_libraries(${LIB_NAME} PUBLIC stdc++fs)
elseif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 9.1))
    target_link_libraries(${LIB_NAME} PUBLIC stdc++fs)
endif()
# Add an alias so that the library can be used inside the build tree, e.g. when testing.
add_library(SRL::${PROJECT_NAME} ALIAS ${LIB_NAME})
# This is required so that the exported target has the name Supereight2 and not supereight.
set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME ${PROJECT_NAME})

# Compile the tests

# Compile the app

# Install the library
set(PACKAGE_NAME ${PROJECT_NAME})
string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
# The directory where the .cmake files will be installed
include(GNUInstallDirs)
if(WIN32 AND NOT CYGWIN)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
    set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()
# Generate *Config.cmake and *ConfigVersion.cmake files
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/${PACKAGE_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
    INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
# Install files
install(DIRECTORY include/se ${CMAKE_CURRENT_BINARY_DIR}/include/se DESTINATION include PATTERN "*.in" EXCLUDE)
install(TARGETS ${LIB_NAME} EXPORT ${PACKAGE_NAME}-targets DESTINATION lib)
install(EXPORT ${PACKAGE_NAME}-targets
    FILE ${PACKAGE_NAME}Targets.cmake
    NAMESPACE SRL::
    DESTINATION ${INSTALL_CMAKEDIR})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake
    DESTINATION ${INSTALL_CMAKEDIR})
