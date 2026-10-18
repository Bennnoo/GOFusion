/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portable RGB input contract for the Gaussian-mapping ingest path.
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace gs {

enum class RgbInputMemoryKind {
    OrdinaryHost,
    RegisteredMapped,
    PinnedStaging,
};

enum class RgbChannelOrder {
    RGB,
    BGR,
};

struct RgbInputView {
    const uint8_t* host_ptr = nullptr;
    const uint8_t* mapped_device_ptr = nullptr;
    int width = 0;
    int height = 0;
    size_t row_stride = 0;
    RgbInputMemoryKind memory_kind = RgbInputMemoryKind::OrdinaryHost;
    RgbChannelOrder channel_order = RgbChannelOrder::RGB;
    std::shared_ptr<void> owner;
    std::function<void(void*)> record_cuda_use;
    std::function<void()> release_owner;

    bool valid() const noexcept
    {
        return host_ptr && width > 0 && height > 0
               && row_stride >= static_cast<size_t>(width) * 3u;
    }
};

inline const char* rgbInputMemoryKindName(const RgbInputMemoryKind kind) noexcept
{
    switch (kind) {
        case RgbInputMemoryKind::RegisteredMapped:
            return "registered-mapped";
        case RgbInputMemoryKind::PinnedStaging:
            return "pinned-staging";
        case RgbInputMemoryKind::OrdinaryHost:
        default:
            return "ordinary-host";
    }
}

} 
