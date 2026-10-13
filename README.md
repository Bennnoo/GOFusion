<!-- SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications) -->

<h1 align="center">GOFusion</h1>

<p align="center">
  <a href="#build">Build</a> · <a href="#datasets">Datasets</a> · <a href="#usage">Usage</a> · <a href="#license">License</a>
</p>

## Build

Linux x86-64 requires a CUDA-capable NVIDIA GPU, a compatible NVIDIA driver, CMake 3.24 or newer, and a CUDA-enabled **C++11 ABI** LibTorch distribution. CUDA, LibTorch and the selected GPU architecture must be compatible.

The build uses bundled BRISK, DBoW2, OpenGV and Ceres sources. Tested on Ubuntu 24.04 with CUDA 11.8, LibTorch 2.5.1 with CUDA 11.8, OpenCV 4.8, GCC 12 for C++ and GCC 11 as the CUDA host compiler.

### Dependencies

On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential gcc-12 g++-12 gcc-11 g++-11 \
  cmake ninja-build pkg-config git curl unzip \
  libeigen3-dev libgoogle-glog-dev libgflags-dev libboost-all-dev \
  libsuitesparse-dev liblapack-dev libblas-dev libtbb-dev \
  libpcl-dev libgeographiclib-dev libzstd-dev \
  libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev \
  python3
```

Install CUDA Toolkit 11.8 for your Linux distribution, then the CUDA 11.8 / C++11 ABI LibTorch archive (skip if already installed):

```bash
mkdir -p "$HOME/dep/downloads"
curl -fL -C - \
  'https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip' \
  -o "$HOME/dep/downloads/libtorch-cu118.zip"
# Use an empty destination directory.
mkdir -p "$HOME/dep/libtorch-cu118-unpack"
unzip -q "$HOME/dep/downloads/libtorch-cu118.zip" -d "$HOME/dep/libtorch-cu118-unpack"
mv "$HOME/dep/libtorch-cu118-unpack/libtorch" "$HOME/dep/libtorch_cu118"
```

The installed directory must contain `share/cmake/Torch/TorchConfig.cmake`.

OpenCV 4.8 or newer is required. If needed, build OpenCV 4.8.0:

```bash
mkdir -p "$HOME/dep/src"
cd "$HOME/dep/src"
curl -fL https://github.com/opencv/opencv/archive/refs/tags/4.8.0.tar.gz -o opencv-4.8.0.tar.gz
tar -xzf opencv-4.8.0.tar.gz
env -i HOME="$HOME" PATH=/usr/bin:/bin \
  /usr/bin/cmake --fresh -S opencv-4.8.0 -B opencv-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/dep/opencv-4.8" \
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,calib3d,features2d,flann \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF
env -i HOME="$HOME" PATH=/usr/bin:/bin /usr/bin/cmake --build opencv-build --parallel 6
env -i HOME="$HOME" PATH=/usr/bin:/bin /usr/bin/cmake --install opencv-build
```

### Configure and build

Use the clean environment below even when Conda is inactive: its `bin/` may remain in `PATH` and expose incompatible dependencies. These commands also exclude shell-provided ROS paths without changing your shell settings. Dependency paths must be outside Conda.

`--fresh` clears the CMake cache and regenerates the configuration. Use it on first configure or after a failed configure/build with mixed dependencies. Existing object files remain and are rebuilt as needed.

```bash
cd /path/to/GOFusion_release

export CUDA_HOME=/usr/local/cuda-11.8
export LIBTORCH_ROOT="$HOME/dep/libtorch_cu118"
export OPENCV_ROOT="$HOME/dep/opencv-4.8"

env -i HOME="$HOME" PATH="$CUDA_HOME/bin:/usr/bin:/bin" \
  /usr/bin/cmake --fresh -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT;$OPENCV_ROOT" \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
  -DCMAKE_IGNORE_PREFIX_PATH="$HOME/anaconda3;$HOME/miniconda3;$HOME/miniforge3;$HOME/mambaforge"

env -i HOME="$HOME" PATH="$CUDA_HOME/bin:/usr/bin:/bin" \
  /usr/bin/cmake --build build --target okvis2x_app_synchronous --parallel 22
```

For incremental builds, repeat only `cmake --build` above with the same clean environment.

Adjust the three dependency paths. `native` selects the GPU visible at configure time; for another GPU, pass its CUDA compute capability, e.g. `86` for RTX 30-series or `89` for RTX 4090. Reduce parallelism if host memory is limited. Use a fresh build directory when changing compilers or LibTorch ABI.

Keep `build/okvis2x_app_synchronous` and its loop-closure vocabulary `build/small_voc.yml.gz` together. The build records dependency library paths for execution.

## Datasets

### uHumans2

Download and prepare the apartment sequence:

```bash
bash scripts/bash/downloads/download_uhumans2_apartment.sh
```

The launcher accepts an **extracted** `uHumans2_apartment_s1_00h` directory. Image dimensions, calibration and units must match `config/uhumans2` (720 × 480 in the supplied configuration).

```text
uHumans2_apartment_s1_00h/
  cameras/left_mono/images/
  cameras/right_mono/images/
  cameras/left_rgb/images/
  cameras/depth/depth_mm/
  stereo/mono_pairs.csv
  rgbd/left_rgb_depth_pairs.csv
  imu/noisy.csv
```

Required CSV columns:

| File | Columns |
| --- | --- |
| `stereo/mono_pairs.csv` | `timestamp_ns,left_mono_file,right_mono_file` |
| `rgbd/left_rgb_depth_pairs.csv` | `timestamp_ns,left_rgb_file,depth_mm_file` |
| `imu/noisy.csv` | `timestamp_ns,angular_velocity_x,angular_velocity_y,angular_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z` |


### TUM RGB-D

Download and prepare all three sequences under `data/tum_rgbd/`:

```bash
bash scripts/bash/downloads/download_tum_rgbd.sh
```

## Usage

### uHumans2

The default input is `data/uhumans2/apartment_scene/uHumans2_apartment_s1_00h`. Add `--headless` to measure runtime without GUI overhead:

```bash
bash scripts/bash/datasets/uhumans2/run_uhumans2_apartment_s1_00h.sh --headless
```

To use an existing extracted sequence:

```bash
bash scripts/bash/datasets/uhumans2/run_uhumans2_apartment_s1_00h.sh \
  /path/to/uHumans2_apartment_s1_00h
```

### TUM RGB-D

Mapping with ground-truth poses:

```bash
bash scripts/bash/datasets/tum_rgbd/run_tum_rgbd_gtpose.sh rgbd_dataset_freiburg1_desk
```

Change the sequence name or the launcher's `SEQUENCE` default, or pass an existing sequence directory:

```bash
bash scripts/bash/datasets/tum_rgbd/run_tum_rgbd_gtpose.sh /path/to/rgbd_dataset_freiburg1_desk
```

## License

GOFusion includes components from [GSFusion](https://github.com/ethz-mrl/GSFusion), [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X), [MonoGS](https://github.com/muskie82/MonoGS) and [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting).

The combined application is restricted to **non-commercial research and evaluation** by its Gaussian-Splatting components. See [License](LICENSE) and [Third-party notices](THIRD_PARTY_NOTICES.md) for component terms and redistribution requirements.
