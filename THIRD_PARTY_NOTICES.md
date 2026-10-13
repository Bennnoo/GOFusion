<!-- SPDX-FileCopyrightText: 2026 Anonymous GOFusion author -->

# Third-party notices

GOFusion contains modified upstream components. Links below identify the original authors' upstream sources; bundled license texts and source notices retain their terms.

## Main components

| Upstream | Included code | License |
| --- | --- | --- |
| [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X) | `okvis_*`, supporting CMake files and vocabulary | [BSD-3-Clause](LICENSES/OKVIS2-X-BSD-3-Clause.txt), with dependency-specific exceptions below |
| [Supereight2](https://github.com/ethz-mrl/supereight2) | Volumetric mapping in `supereight-2-srl/include/se` and `src/{common,image,map,sensor}` | [BSD-3-Clause](supereight-2-srl/LICENSES/BSD-3-Clause.txt) for the core; [CC0-1.0](supereight-2-srl/LICENSES/CC0-1.0.txt) for CC0-marked build files; [MIT](supereight-2-srl/LICENSES/MIT.txt) for MIT-marked helpers; [copyright metadata](supereight-2-srl/.reuse/dep5) |
| [GSFusion](https://github.com/ethz-mrl/GSFusion) | Gaussian integration, quad-tree, model and rendering utilities in `supereight-2-srl/include/{se,gs}` and `src/gs` | [BSD-3-Clause](supereight-2-srl/LICENSES/BSD-3-Clause.txt) for BSD-marked files; [Gaussian-Splatting](supereight-2-srl/LICENSES/Gaussian-Splatting-License.md) for the Gaussian model and rendering derivatives |
| [MonoGS](https://github.com/muskie82/MonoGS): [diff-gaussian-rasterization-w-pose](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose) | `supereight-2-srl/mono_cuda_rasterizer`, `include/gs/monogs_rasterize_points.cuh`, `src/gs/monogs_rasterize_points.cu` | [Gaussian-Splatting](supereight-2-srl/mono_cuda_rasterizer/LICENSE.md); Faster-GS portions and NVIDIA helper retain their separate terms |
| [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) | `supereight-2-srl/fastergs_cuda_rasterizer`, `include/gs/fastergs_rasterizer.cuh`, `src/gs/fastergs_rasterizer.cu`; adapted bucket rendering in `mono_cuda_rasterizer/src/{forward,backward}.cu` | [Apache-2.0](supereight-2-srl/fastergs_cuda_rasterizer/LICENSE) for these portions; NVIDIA helper exception below |

OKVIS2-X credits Autonomous Systems Lab / ETH Zurich (2015), Smart Robotics Lab / Imperial College London (2020), and Mobile Robotics Lab / Technical University of Munich and ETH Zurich (2025). GSFusion credits Jiaxin Wei and Smart Robotics Lab / Technical University of Munich; its Gaussian code also retains Janusch Patas's notices. Supereight2 retains its per-file author notices. The original Gaussian-Splatting software is owned by Inria and MPII. Faster-GS is copyright 2026 The Faster-GS Authors.


## Other bundled components

| Component | License and notices |
| --- | --- |
| [BRISK / AGAST](https://github.com/ethz-mrl/brisk/blob/1ef8b42a5c2fdd0e0c976f3ab7b179806381f570/LICENSE) | [BSD terms](external/brisk/LICENSE) and individual source notices |
| [DBoW2](https://github.com/dorian3d/DBoW2/blob/3924753db6145f12618e7de09b7e6b258db93c6e/LICENSE.txt) | [Modified BSD terms](external/DBoW2/LICENSE.txt), including author notification on redistribution |
| [OpenGV](https://github.com/laurentkneip/opengv/blob/91f4b19c73450833a40e463ad3648aae80b3a7f3/include/opengv/types.hpp) | [BSD notice in source](external/opengv/include/opengv/types.hpp) |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver/blob/85331393dc0dff09f6fb9903ab0c4bfa3e134b01/LICENSE) | [BSD, Apache-2.0 and libmv MIT notices](external/ceres-solver/LICENSE) |
| [GoogleTest / GoogleMock](https://github.com/google/googletest/blob/58d77fa8070e8cec2dc1ed015d66b454c8d78850/LICENSE) | [OKVIS copy](external/googletest/LICENSE), [Supereight copy](supereight-2-srl/third_party/googletest/LICENSE); [generator upstream](https://github.com/google/googletest/blob/e2239ee6043f73722e7aa812a459f54a28552929/googlemock/scripts/generator/LICENSE) and [bundled license](supereight-2-srl/third_party/googletest/googlemock/scripts/generator/LICENSE) |
| [pybind11](https://github.com/pybind/pybind11/blob/v2.2.4/LICENSE) | [BSD-style license and exceptions](external/opengv/python/pybind11/LICENSE); [LLVM/Clang upstream](https://github.com/llvm/llvm-project/blob/llvmorg-3.2.0/clang/LICENSE.TXT) and [bundled tooling license](external/opengv/python/pybind11/tools/clang/LICENSE.TXT) |
| [SRL projection](https://github.com/ethz-mrl/srl-projection/blob/2f1de25fff2a5f4351344a60cd5ee8fc42f971c7/include/srl/projection/PinholeCamera.hpp) | [BSD notices in headers](supereight-2-srl/third_party/srl_projection/include/srl/projection/PinholeCamera.hpp) and [copyright metadata](supereight-2-srl/.reuse/dep5) |
| [nlohmann/json](https://github.com/nlohmann/json/blob/5d2754306d67d1e654a1a34e1d2e74439a9d53b3/LICENSE.MIT) | [MIT](supereight-2-srl/third_party/json/LICENSE.MIT) and per-file notices, including embedded third-party code; `detail/output/{binary_writer,output_adapters,serializer}.hpp` come from [v3.11.3](https://github.com/nlohmann/json/tree/v3.11.3/include/nlohmann/detail/output), with the same [license text](https://github.com/nlohmann/json/blob/v3.11.3/LICENSE.MIT) |
| [GLM](https://github.com/g-truc/glm/blob/5c46b9c07008ae65cb81ab79cd677ecc1934b903/copying.txt) | [MIT option of the upstream license](supereight-2-srl/third_party/glm/copying.txt) |
| [tinyply](https://github.com/ddiakopoulos/tinyply/blob/40aa4a0ae9e9c203e11893f78b8bcaf8a50e65f0/source/tinyply.h) | [Public-domain dedication and fallback permission](supereight-2-srl/third_party/tinyply/source/tinyply.h) |
| [tinycolormap](https://github.com/yuki-koyama/tinycolormap/blob/master/include/tinycolormap.hpp) / [Turbo palette](https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a) | [Yuki Koyama's MIT and Google LLC's Turbo notices](supereight-2-srl/include/se/external/tinycolormap.hpp); [Apache-2.0 text](LICENSES/Apache-2.0.txt) |
| [PyTorch SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt) | `gaussian` / `create_window` in `include/gs/gaussian_utils.cuh` and `ssim` in `include/gs/loss_utils.cuh`, under `supereight-2-srl`: [Evan Su, 2017, MIT](LICENSES/pytorch-ssim-MIT.txt), inherited through the Gaussian-Splatting utilities |
| [NVIDIA helper math](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h) | Copyright (c) 2022 NVIDIA CORPORATION; BSD notices in [MonoGS helper](supereight-2-srl/mono_cuda_rasterizer/include/mono_helper_math.h) and [Faster-GS helper](supereight-2-srl/fastergs_cuda_rasterizer/include/fastergs/helper_math.h) |

## Redistribution

- The combined application is for **non-commercial research and evaluation** under the Gaussian-Splatting license. Commercial use requires permission from the relevant rights holders.
- Retain all applicable license texts, copyright, attribution and disclaimer notices. Binary distributions must include those materials as well, the current build does not assemble a license bundle automatically.
- Apache-2.0 derivatives must identify modified files. The adapted Faster-GS sources carry those notices. Do not imply endorsement by upstream authors.
- **Before redistributing source or binaries containing DBoW2, notify its original author, Dorian Galvez-Lopez**, as required by [clause 3](external/DBoW2/LICENSE.txt). This project's notification has not yet been confirmed. Its license also requests the included citation for academic use.

## Linked dependencies

System libraries and downloaded dependencies are not relicensed by GOFusion. Binary packages and container images must account for their licenses and transitive dependencies.

**Binary redistribution compatibility remains unresolved:** Ceres enables SuiteSparse by default, and the inspected Ubuntu 24.04 build links CHOLMOD from SuiteSparse 7.6.1. [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.6.1/CHOLMOD/Doc/License.txt) includes GPL-2.0-or-later modules; [SPQR](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.6.1/SPQR/Doc/License.txt), also selected by the Ceres configuration, is GPL-2.0-or-later. Those terms conflict with the Gaussian-Splatting non-commercial restriction when distributing a combined work. Resolve the combination through compatible dependencies or appropriate additional permissions before distributing binaries. Retaining license texts alone does not resolve this issue.

## Upstream license references

- OKVIS2-X: [0046491 — initial release](https://github.com/ethz-mrl/OKVIS2-X/tree/0046491d61dd5a8e915810523e0b427165ab97a9), [BSD-3-Clause](https://github.com/ethz-mrl/OKVIS2-X/blob/0046491d61dd5a8e915810523e0b427165ab97a9/LICENSE).
- Supereight2: [f98c564](https://github.com/ethz-mrl/supereight2/tree/f98c56405e83605334da691f22a88dae84ffe6f1), [license texts](https://github.com/ethz-mrl/supereight2/tree/f98c56405e83605334da691f22a88dae84ffe6f1/LICENSES) and [copyright metadata](https://github.com/ethz-mrl/supereight2/blob/f98c56405e83605334da691f22a88dae84ffe6f1/.reuse/dep5).
- GSFusion: [b9a6711](https://github.com/ethz-mrl/GSFusion/tree/b9a671135a21a24a486342836372746c4ea4adad), [BSD-3-Clause](https://github.com/ethz-mrl/GSFusion/blob/b9a671135a21a24a486342836372746c4ea4adad/LICENSES/BSD-3-Clause.txt) and [Gaussian-Splatting](https://github.com/ethz-mrl/GSFusion/blob/b9a671135a21a24a486342836372746c4ea4adad/LICENSES/Gaussian-Splatting-License.md).
- MonoGS: [6c9254c — main-code agreement](https://github.com/muskie82/MonoGS/blob/6c9254c319d8bff5caeef65259e6bb0941a9b9f6/LICENSE.md); included rasterizer: [43e21bf — Gaussian-Splatting](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/blob/43e21bff91cd24986ee3dd52fe0bb06952e50ec7/LICENSE.md).
- Faster-GS: [c09cf3d — Apache-2.0](https://github.com/nerficg-project/faster-gaussian-splatting/blob/c09cf3d029941f441c8d79fb8c63550b4d8ed76f/LICENSE).

The original [Inria / MPII Gaussian-Splatting license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md) also records the SSIM attribution in section 6. The standard [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0) is published by the Apache Software Foundation. External dependencies and downloaded datasets retain their own terms.
