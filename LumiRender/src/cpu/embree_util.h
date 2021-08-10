//
// Created by Zero on 2021/8/10.
//


#pragma once

#include <embree3/rtcore.h>
#include "base_libs/geometry/util.h"

namespace luminous {
    inline namespace cpu {
        NDSC_INLINE RTCRay to_RTCRay(Ray r) {
            RTCRay ray;
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
    }
}