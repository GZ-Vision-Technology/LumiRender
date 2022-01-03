//
// Created by Zero on 2021/5/5.
//


#pragma once

#include "base_libs/math/common.h"
#include "interaction.h"

#if defined(__CUDACC__)
#include "gpu/shaders/optix_util.h"
#else
#include "cpu/accel/embree_util.h"
#endif
namespace luminous {
    inline namespace render {

        LM_XPU_INLINE bool intersect_closest(uint64_t traversable_handle, Ray ray, HitInfo *hit_info) {
#if defined(__CUDACC__)
            return traceClosestHit((OptixTraversableHandle)traversable_handle, ray, hit_info);
#else
            return rtc_intersect((RTCScene)traversable_handle, ray, hit_info);
#endif
        }

        LM_XPU_INLINE bool intersect_any(uint64_t traversable_handle, Ray ray) {
#if defined(__CUDACC__)
            return traceAnyHit((OptixTraversableHandle)traversable_handle, ray);
#else
            return rtc_occlusion((RTCScene)traversable_handle, ray);
#endif
        }

    } // luminous::render
} // luminous
