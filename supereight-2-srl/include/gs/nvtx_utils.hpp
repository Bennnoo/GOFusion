/*
 * SPDX-FileCopyrightText: 2026 Boxin Bu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdlib>
#include <cstring>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GOFUSION_HAS_NVTX 1
#else
#define GOFUSION_HAS_NVTX 0
#endif

namespace gs::profiling {

inline bool nvtxEnabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("GOFUSION_NVTX");
        if (!value || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0
               && std::strcmp(value, "false") != 0
               && std::strcmp(value, "FALSE") != 0
               && std::strcmp(value, "off") != 0
               && std::strcmp(value, "OFF") != 0;
    }();
    return enabled;
}

class NvtxRange {
    public:
    explicit NvtxRange(const char* name) : active_(nvtxEnabled())
    {
#if GOFUSION_HAS_NVTX
        if (active_) {
            nvtxRangePushA(name);
        }
#else
        (void) name;
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

    ~NvtxRange()
    {
        stop();
    }

    void stop()
    {
#if GOFUSION_HAS_NVTX
        if (active_) {
            nvtxRangePop();
            active_ = false;
        }
#endif
    }

    private:
    bool active_ = false;
};

} 

#define GOFUSION_NVTX_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define GOFUSION_NVTX_CONCAT(lhs, rhs) GOFUSION_NVTX_CONCAT_IMPL(lhs, rhs)
#define GOFUSION_NVTX_RANGE(name) \
    ::gs::profiling::NvtxRange GOFUSION_NVTX_CONCAT(gofusion_nvtx_range_, __COUNTER__)(name)
