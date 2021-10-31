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

        LM_XPU_INLINE bool intersect_closest(uint64_t traversable_handle, Ray ray, HitContext *hit_ctx) {
#if defined(__CUDACC__)
            return traceClosestHit((OptixTraversableHandle)traversable_handle, ray, hit_ctx);
#else
            return rtc_intersect((RTCScene)traversable_handle, ray, hit_ctx);
#endif
        }

        LM_XPU_INLINE bool intersect_any(uint64_t traversable_handle, Ray ray) {
#if defined(__CUDACC__)
            return traceAnyHit((OptixTraversableHandle)traversable_handle, ray);
#else
            return rtc_occlusion((RTCScene)traversable_handle, ray);
#endif
        }

        class VisibilityTester {
        private:
            Interaction _p0{}, _p1{};
        public:
            LM_XPU VisibilityTester() = default;

            LM_XPU VisibilityTester(const Interaction &p0,
                                    const Interaction &p1)
                    : _p0(p0), _p1(p1) {}

            LM_ND_XPU const Interaction &P0() const { return _p0; }

            LM_ND_XPU const Interaction &P1() const { return _p1; }

            LM_ND_XPU bool occluded(uint64_t traversable_handle) const {
                auto ray = _p0.spawn_ray_to(_p1);
                return intersect_any(traversable_handle, ray);
            }
        };

    } // luminous::render
} // luminous
