//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "soa.h"
#include "base_libs/header.h"
#include "core/backend/device.h"
#include "base_libs/geometry/common.h"
#include "base_libs/math/common.h"
#include "base_libs/optics/rgb.h"
#include "render/bxdfs/bsdf.h"
#include "render/lights/light_util.h"
#include "work_queue.h"
#include <cuda.h>

namespace luminous {

    inline namespace render {

#if 0 // unfold

        LUMINOUS_SOA(int3, x, y, z)
        template<>
        struct SOA<luminous::int3> {
        public:
            static constexpr bool definitional = true;
            using element_type = luminous::int3;
            SOA() = default;
            int capacity;
            SOAMember<decltype(element_type::x), Device *>::type x;
            SOAMember<decltype(element_type::y), Device *>::type y;
            SOAMember<decltype(element_type::z), Device *>::type z;
            SOA(int n, Device *device) : capacity(n) {
                x = SOAMember<decltype(element_type::x), Device *>::create(n, device);
                y = SOAMember<decltype(element_type::y), Device *>::create(n, device);
                z = SOAMember<decltype(element_type::z), Device *>::create(n, device);
            }
            SOA &operator=(const SOA &s) {
                capacity = s.capacity;
                this->x = s.x;
                this->y = s.y;
                this->z = s.z;
                return *this;
            }
            element_type operator[](int i) const {
                (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (21)), 0));;;
                element_type r;
                r.x = this->x[i];
                r.y = this->y[i];
                r.z = this->z[i];
                return r;
            }
            struct GetSetIndirector {
                SOA *soa;
                int i;
                operator element_type() const {
                    element_type r;
                    r.x = soa->x[i];
                    r.y = soa->y[i];
                    r.z = soa->z[i];
                    return r;
                }
                void operator=(const element_type &a) const {
                    soa->x[i] = a.x;
                    soa->y[i] = a.y;
                    soa->z[i] = a.z;
                }
            };
            GetSetIndirector operator[](int i) {
                (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (21)), 0));;;
                return GetSetIndirector{this, i};
            }
        };

#endif

        LUMINOUS_SOA(float2, x, y)

        LUMINOUS_SOA(float3, x, y, z)

        LUMINOUS_SOA(float4, x, y, z, w)

        LUMINOUS_SOA(Spectrum, x, y, z)

        LUMINOUS_SOA(Ray, org_x, org_y, org_z, dir_x, dir_y, dir_z, t_max)

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
            RayQueue(int n, Device *device)
                    : WorkQueue<RayWorkItem>(n, device) {}

            RayQueue(const RayQueue &other)
                    : WorkQueue<RayWorkItem>(other) {}

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

        class Light;

        struct HitAreaLightWorkItem {
            Light *light;
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

        struct MaterialEvalWorkItem {
            float3 pos;
            float3 ng;
            float3 ns;
            float2 uv;
            float3 wo;
            bool any_non_specular_bounces;
            int pixel_index;
            Spectrum throughput;
        };

        LUMINOUS_SOA(MaterialEvalWorkItem, pos, ng, ns, uv, wo,
                     any_non_specular_bounces, pixel_index, throughput)

        using MaterialEvalQueue = WorkQueue<MaterialEvalWorkItem>;

        struct PixelSampleState {
            uint2 pixel;
            Spectrum L;
            float filter_weight;
            Spectrum sensor_ray_weight;
            RaySamples samples;
        };

        LUMINOUS_SOA(PixelSampleState, pixel, L, filter_weight, sensor_ray_weight, samples)
    }
}