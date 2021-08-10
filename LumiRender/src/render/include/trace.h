//
// Created by Zero on 2021/5/5.
//


#pragma once

#include "base_libs/math/common.h"
#include "interaction.h"

#if defined(__CUDACC__)
#include "gpu/shaders/optix_util.h"
#else
#include "cpu/embree_util.h"
#endif
namespace luminous {
    inline namespace render {

        XPU_INLINE bool intersect_closest(uint64_t traversable_handle, Ray ray, PerRayData *prd) {
#if defined(__CUDACC__)
            return traceRadiance((OptixTraversableHandle)traversable_handle, ray, prd);
#else
            // CPU is not implemented
            RTCIntersectContext context;
            RTCRayHit rh = to_RTCRayHit(ray);
            rtcIntersect1(reinterpret_cast<RTCScene>(traversable_handle), &context, &rh);
            prd->closest_hit.instance_id = rh.hit.instID[0];
            prd->closest_hit.triangle_id = rh.hit.primID;
            prd->closest_hit.bary = make_float2(rh.hit.u, rh.hit.v);
            return prd->is_hit();
#endif
        }

        XPU_INLINE bool intersect_any(uint64_t traversable_handle, Ray ray) {
#if defined(__CUDACC__)
            return traceOcclusion((OptixTraversableHandle)traversable_handle, ray);
#else
            // CPU is not implemented
            assert(0);
            return false;
#endif
        }

        class VisibilityTester {
        private:
            Interaction _p0, _p1;
        public:
            XPU VisibilityTester() = default;

            XPU VisibilityTester(const Interaction &p0,
                                 const Interaction &p1)
                    : _p0(p0), _p1(p1) {}

            NDSC_XPU const Interaction &P0() const { return _p0; }

            NDSC_XPU const Interaction &P1() const { return _p1; }

            NDSC_XPU bool occluded(uint64_t traversable_handle) const {
                auto ray = _p0.spawn_ray_to(_p1);
                return intersect_any(traversable_handle, ray);
            }
        };

    } // luminous::render
} // luminous
