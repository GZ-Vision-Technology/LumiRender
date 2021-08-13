//
// Created by Zero on 2021/8/10.
//


#pragma once

#include <embree3/rtcore.h>
#include "render/include/interaction.h"
#include "base_libs/geometry/util.h"

namespace luminous {
    inline namespace cpu {
        NDSC_INLINE RTCRay to_RTCRay(Ray r) {
            RTCRay ray{};
            ray.org_x = r.org_x;
            ray.org_y = r.org_y;
            ray.org_z = r.org_z;
            ray.dir_x = r.dir_x;
            ray.dir_y = r.dir_y;
            ray.dir_z = r.dir_z;
            ray.tnear = r.t_min;
            ray.tfar  = r.t_max;
            ray.flags = 0;
            return ray;
        }

        NDSC_INLINE RTCRayHit to_RTCRayHit(Ray ray) {
            RTCRay rtc_ray = to_RTCRay(ray);
            RTCRayHit rh{};
            rh.ray = rtc_ray;
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
            return rh;
        }

        NDSC_INLINE RTCBounds to_RTCBounds(Box3f box) {
            return RTCBounds{box.lower.x,box.lower.y,box.lower.z,0,
                             box.upper.x,box.upper.y,box.upper.z,0};
        }

        NDSC_INLINE bool rtc_intersect(RTCScene scene, Ray ray, PerRayData *prd) {
            RTCIntersectContext context{};
            rtcInitIntersectContext(&context);
            RTCRayHit rh = to_RTCRayHit(ray);
            rtcIntersect1(scene, &context, &rh);
            prd->closest_hit.instance_id = rh.hit.instID[0];
            prd->closest_hit.triangle_id = rh.hit.primID;
            prd->closest_hit.bary = make_float2(rh.hit.u, rh.hit.v);
            return prd->is_hit();
        }

        NDSC_INLINE bool rtc_occlusion(RTCScene scene, Ray ray) {
            RTCIntersectContext context{};
            rtcInitIntersectContext(&context);
            RTCRay rtc_ray = to_RTCRay(ray);
            rtcOccluded1(scene, &context, &rtc_ray);
            return rtc_ray.tfar < 0;
        }
    }
}