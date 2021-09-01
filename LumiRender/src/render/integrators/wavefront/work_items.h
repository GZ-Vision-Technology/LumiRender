//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "soa.h"
#include "base_libs/header.h"
#include "core/backend/device.h"
#include "base_libs/geometry/common.h"
#include "base_libs/math/common.h"
#include "render/bxdfs/bsdf.h"
#include "render/lights/light.h"
#include "work_queue.h"
#include <cuda.h>

namespace luminous {

    inline namespace render {

        LUMINOUS_SOA(float2, x, y)

        LUMINOUS_SOA(float3, x, y, z)

        LUMINOUS_SOA(float4, x, y, z, w)

        LUMINOUS_SOA(Ray, org_x, org_y, org_z, dir_x, dir_y, dir_z, t_max)

        LUMINOUS_SOA(BSDF, _bxdf, _ng, _shading_frame)

        enum RaySampleFlag {
            hasMedia = 1 << 0,
            hasSubsurface = 1 << 1
        };

        struct RaySamples {
            struct {
                float2 u{};
                float uc{};
            } direct;
            struct {
                float2 u{};
                float uc{}, rr{};
            } indirect;
            RaySampleFlag flag;
        };

        LUMINOUS_SOA(RaySamples, direct, indirect, flag)

        struct RayWorkItem {
            Ray ray;
            int depth;
            int pixel_index;
            Spectrum throughput;
            LightSampleContext prev_lsc;
            float eta_scale;
            bool specular_bounce;
            bool any_non_specular_bounces;
        };

        LUMINOUS_SOA(RayWorkItem, ray, depth, pixel_index, throughput,
                     prev_lsc, eta_scale, specular_bounce,
                     any_non_specular_bounces)

        class RayQueue : public WorkQueue<RayWorkItem> {
        public:
            NDSC_XPU_INLINE int push_primary_ray(const Ray &ray, int pixel_index) {
                int index = allocate_entry();
                this->ray[index] = ray;
                this->depth[index] = 0;
                this->pixel_index[index] = pixel_index;
                this->throughput[index] = Spectrum(1.f);
                this->eta_scale[index] = 1.f;
                this->any_non_specular_bounces[index] = false;
                this->specular_bounce[index] = false;
                return index;
            }

            NDSC_XPU_INLINE int push_secondary_ray(const Ray &ray, int depth, const LightSampleContext &prev_lsc,
                                                   const Spectrum &throughput, float eta_scale, bool specular_bounce,
                                                   bool any_non_specular_bounces, int pixel_index) {
                int index = allocate_entry();
                this->ray[index] = ray;
                this->depth[index] = depth;
                this->pixel_index[index] = pixel_index;
                this->prev_lsc[index] = prev_lsc;
                this->throughput[index] = throughput;
                this->any_non_specular_bounces[index] = any_non_specular_bounces;
                this->specular_bounce[index] = specular_bounce;
                this->eta_scale[index] = eta_scale;
                return index;
            }
        };

        struct EscapedRayWorkItem {
            float3 ray_o;
            float3 ray_d;
            int depth;
            int pixel_index;
            Spectrum throughput;
            int specular_bounce;
            LightSampleContext prev_lsc;
        };

        LUMINOUS_SOA(EscapedRayWorkItem, ray_o, ray_d, depth, pixel_index,
                     throughput, specular_bounce, prev_lsc)

        class EscapedRayQueue : public WorkQueue<EscapedRayWorkItem> {
        public:
            using WorkQueue::WorkQueue;

            NDSC_XPU_INLINE int push(RayWorkItem r) {
                EscapedRayWorkItem item{r.ray.origin(), r.ray.direction(), r.depth,
                                        r.pixel_index, r.throughput, r.specular_bounce, r.prev_lsc};
                return WorkQueue::push(item);
            }
        };

        struct HitAreaLightWorkItem {
            Light light;
            float3 pos;
            float3 ng;
            float2 uv;
            float3 wo;
            int depth;
            Spectrum throughput;
            LightSampleContext prev_lsc;
            int specular_bounce;
            int pixel_index;
        };

        LUMINOUS_SOA(HitAreaLightWorkItem, light, pos, ng, uv, wo, depth,
                     throughput, prev_lsc, specular_bounce, pixel_index)

        using HitAreaLightQueue = WorkQueue<HitAreaLightWorkItem>;

        struct ShadowRayWorkItem {
            Ray ray;
            Spectrum Ld;
            int pixel_index;
        };

        LUMINOUS_SOA(ShadowRayWorkItem, ray, Ld, pixel_index)

        using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;
    }
}