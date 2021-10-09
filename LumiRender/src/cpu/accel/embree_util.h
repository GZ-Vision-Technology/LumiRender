//
// Created by Zero on 2021/8/10.
//


#pragma once

#include <embree3/rtcore.h>
#include "render/include/interaction.h"
#include "base_libs/geometry/util.h"

using std::cout;
using std::endl;

namespace luminous {
    inline namespace cpu {
        LM_ND_INLINE RTCRay to_RTCRay(Ray r) {
            RTCRay ray{};
            ray.org_x = r.org_x;
            ray.org_y = r.org_y;
            ray.org_z = r.org_z;
            ray.dir_x = r.dir_x;
            ray.dir_y = r.dir_y;
            ray.dir_z = r.dir_z;
            ray.tnear = 0;
            ray.tfar = r.t_max;
            ray.flags = 0;
            return ray;
        }

        LM_ND_INLINE RTCRayHit to_RTCRayHit(Ray ray) {
            RTCRay rtc_ray = to_RTCRay(ray);
            RTCRayHit rh{};
            rh.ray = rtc_ray;
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
            return rh;
        }

        template<int ray_num>
        struct RTCType {
            using RayType = RTCRayNp;
            using HitType = RTCHitNp;
            using RHType = RTCRayHitNp;

            template<typename... Args>
            static auto rtcIntersect(Args &&...args) {
                return rtcIntersectNp(std::forward<Args>(args)..., ray_num);
            }
        };

        template<>
        struct RTCType<4> {
            using RayType = RTCRay4;
            using HitType = RTCHit4;
            using RHType = RTCRayHit4;

            template<typename... Args>
            static auto rtcIntersect(Args &&...args) {
                int a[] = {-1,-1,-1,-1};
                return rtcIntersect4(a, std::forward<Args>(args)...);
            }
        };

        template<>
        struct RTCType<8> {
            using RayType = RTCRay8;
            using HitType = RTCHit8;
            using RHType = RTCRayHit8;

            template<typename... Args>
            static auto rtcIntersect(Args &&...args) {
                int a[] = {-1,-1,-1,-1,-1,-1,-1,-1};
                return rtcIntersect8(a, std::forward<Args>(args)...);
            }
        };

        template<>
        struct RTCType<16> {
            using RayType = RTCRay16;
            using HitType = RTCHit16;
            using RHType = RTCRayHit16;

            template<typename... Args>
            static auto rtcIntersect(Args &&...args) {
                int a[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                return rtcIntersect16(a, std::forward<Args>(args)...);
            }
        };

        template<int ray_num>
        auto to_RTCRayN(const Ray *ray) {
            typename RTCType<ray_num>::RayType ret{};
            // todo allocate memory
            for (int idx = 0; idx < ray_num; ++idx) {
                ret.org_x[idx] = ray[idx].org_x;
                ret.org_y[idx] = ray[idx].org_y;
                ret.org_z[idx] = ray[idx].org_z;
                ret.dir_x[idx] = ray[idx].dir_x;
                ret.dir_y[idx] = ray[idx].dir_y;
                ret.dir_z[idx] = ray[idx].dir_z;
                ret.tnear[idx] = 0;
                ret.tfar[idx] = ray[idx].t_max;
                ret.flags[idx] = 0;
                ret.time[idx] = 0;
            }
            return ret;
        }

        template<int ray_num>
        auto to_RTCRayHitN(const Ray *ray) {
            typename RTCType<ray_num>::RHType ret;
            ret.ray = to_RTCRayN<ray_num>(ray);
            return ret;
        }

        template<int ray_num, typename RHType>
        void to_prd(const RHType rh, PerRayData *prd) {
            static_assert(std::is_same_v<RTCType<ray_num>::RHType, RHType>, "RTCType is not match!");
            for (int i = 0; i < ray_num; ++i) {
                prd[i].hit_point.instance_id = rh.hit.instID[0][i];
                prd[i].hit_point.triangle_id = rh.hit.primID[i];
                prd[i].hit_point.bary.x = 1 - rh.hit.u[i] - rh.hit.v[i];
                prd[i].hit_point.bary.y = rh.hit.u[i];
            }
        }

        template<int ray_num>
        void rtc_intersectNp(RTCScene scene, const Ray *ray, PerRayData *prd) {
            RTCIntersectContext context{};
            rtcInitIntersectContext(&context);
            auto rh = to_RTCRayHitN<ray_num>(ray);
            RTCType<ray_num>::rtcIntersect(scene, &context, &rh);
            to_prd<ray_num>(rh, prd);
        }

        LM_ND_INLINE RTCBounds to_RTCBounds(Box3f box) {
            return RTCBounds{box.lower.x, box.lower.y, box.lower.z, 0,
                             box.upper.x, box.upper.y, box.upper.z, 0};
        }

        LM_ND_INLINE bool rtc_intersect(RTCScene scene, Ray ray, PerRayData *prd) {
            RTCIntersectContext context{};
            rtcInitIntersectContext(&context);
            RTCRayHit rh = to_RTCRayHit(ray);
            rtcIntersect1(scene, &context, &rh);
            prd->hit_point.instance_id = rh.hit.instID[0];
            prd->hit_point.triangle_id = rh.hit.primID;
            float w = 1 - rh.hit.u - rh.hit.v;
            prd->hit_point.bary = make_float2(w, rh.hit.u);
            return prd->is_hit();
        }

        LM_ND_INLINE bool rtc_occlusion(RTCScene scene, Ray ray) {
            RTCIntersectContext context{};
            rtcInitIntersectContext(&context);
            RTCRay rtc_ray = to_RTCRay(ray);
            rtcOccluded1(scene, &context, &rtc_ray);
            return rtc_ray.tfar < 0;
        }
    }
}