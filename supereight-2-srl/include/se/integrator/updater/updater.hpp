/*
 * SPDX-FileCopyrightText: 2021-2024 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2021 Nils Funk
 * SPDX-FileCopyrightText: 2021-2024 Sotiris Papatheodorou
 * SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
 * SPDX-FileCopyrightText: 2024 Jiaxin Wei
 * SPDX-License-Identifier: BSD-3-Clause
 * SPDX-FileCopyrightText: 2026 Boxin Bu (GOFusion modifications)
 */

#ifndef SE_UPDATER_HPP
#define SE_UPDATER_HPP

#include <se/integrator/measurement.hpp>
#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
#include <torch/torch.h>

#include "gs/gaussian.cuh"
#include "gs/gaussian_utils.cuh"
#endif

namespace se {

template<typename MapT, typename SensorT>
struct Updater {
    Updater(MapT& map, const timestamp_t timestamp, const Measurements<SensorT>& measurements);

    Updater(MapT& map,
            std::vector<OctantBase*>& octant_ptrs,
            const timestamp_t timestamp,
            const Measurements<SensorT>& measurements);
};

#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
template<typename MapT, typename SensorT>
struct GSUpdater;
#endif

} // namespace se

#include "multires_ofusion_updater.hpp"
#include "multires_tsdf_updater.hpp"
#include "singleres_tsdf_updater.hpp"
#if defined(OKVIS_ENABLE_GS_MAPPING) && OKVIS_ENABLE_GS_MAPPING
#include "multires_ofusion_gs_updater.hpp"
#endif

#endif // SE_UPDATER_HPP
