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
#include "render/scattering/bsdf_wrapper.h"
#include "render/lights/light_util.h"
#include "work_queue.h"
#include <cuda.h>

namespace luminous {

    inline namespace render {

#if 0 // unfold

        LUMINOUS_SOA(int3, x, y, z)
        template<>
        struct SOA<int3> {
        public:
            static constexpr bool definitional = true;
            using element_type = int3;
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
                (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (23)), 0));;;
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
                (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (23)), 0));;;
                return GetSetIndirector{this, i};
            }
            template<typename TDevice>
            SOA<element_type> to_host(TDevice *device) const {
                DCHECK(device->is_cpu())
                auto ret = SOA<element_type>(capacity, device);
                ret.x = SOAMember<decltype(element_type::x), TDevice *>::clone_to_host(x, capacity, device);
                ret.y = SOAMember<decltype(element_type::y), TDevice *>::clone_to_host(y, capacity, device);
                ret.z = SOAMember<decltype(element_type::z), TDevice *>::clone_to_host(z, capacity, device);
                return ret;
            }
        };

#endif

        LUMINOUS_SOA(float2, x, y)

        LUMINOUS_SOA(float3, x, y, z)

        LUMINOUS_SOA(float4, x, y, z, w)

        LUMINOUS_SOA(Spectrum, x, y, z)

        LUMINOUS_SOA(Ray, org_x, org_y, org_z, dir_x, dir_y, dir_z, t_max)

        LUMINOUS_SOA(HitInfo, instance_id, prim_id, bary)

        LUMINOUS_SOA(LightSampleContext, pos, ng, ns)

        enum RaySampleFlag : uint8_t {
            hasMedia = 1 << 0,
            hasSubsurface = 1 << 1
        };

        struct DirectSamples {
            float2 u{};
            float uc{};
        };

        LUMINOUS_SOA(DirectSamples, u, uc)

        struct IndirectSamples {
            float2 u{};
            float uc{}, rr{};
        };

        LUMINOUS_SOA(IndirectSamples, u, uc, rr)

        struct RaySamples {
            DirectSamples direct;
            IndirectSamples indirect;
            RaySampleFlag flag{};
        };

        LUMINOUS_SOA(RaySamples, direct, indirect, flag)

        struct RayWorkItem {
            Ray ray;
            int depth{};
            int pixel_index{};
            Spectrum throughput;
            LightSampleContext prev_lsc;
            float prev_bsdf_PDF;
            Spectrum prev_bsdf_val;
            float eta_scale{};
        };

        LUMINOUS_SOA(RayWorkItem, ray, depth, pixel_index, throughput,
                     prev_lsc, prev_bsdf_PDF, prev_bsdf_val, eta_scale)

        class RayQueue : public WorkQueue<RayWorkItem> {
        public:
#ifndef __CUDACC__

            RayQueue(int n, Device *device)
                    : WorkQueue<RayWorkItem>(n, device) {}

#endif

            RayQueue(const RayQueue &other)
                    : WorkQueue<RayWorkItem>(other) {}

            RayQueue() = default;

            LM_XPU_INLINE int push_primary_ray(const Ray &ray, int pixel_index) {
                int index = allocate_entry();
                this->ray[index] = ray;
                this->depth[index] = 0;
                this->pixel_index[index] = pixel_index;
                this->throughput[index] = Spectrum(1.f);
                this->eta_scale[index] = 1.f;
                return index;
            }

            LM_XPU_INLINE int push_secondary_ray(const Ray &ray, int depth, const LightSampleContext &prev_lsc,
                                                 const Spectrum &throughput, float bsdf_PDF, Spectrum bsdf_val,
                                                 float eta_scale, int pixel_index) {
                int index = allocate_entry();
                this->ray[index] = ray;
                this->depth[index] = depth;
                this->pixel_index[index] = pixel_index;
                this->prev_lsc[index] = prev_lsc;
                this->prev_bsdf_PDF[index] = bsdf_PDF;
                this->prev_bsdf_val[index] = bsdf_val;
                this->throughput[index] = throughput;
                this->eta_scale[index] = eta_scale;
                return index;
            }
        };

        struct EscapedRayWorkItem {
            float3 ray_o;
            float3 ray_d;
            int depth{};
            int pixel_index{};
            Spectrum throughput;
            LightSampleContext prev_lsc;
            float prev_bsdf_PDF{};
            Spectrum prev_bsdf_val;
        };

        LUMINOUS_SOA(EscapedRayWorkItem, ray_o, ray_d, depth, pixel_index,
                     throughput, prev_lsc, prev_bsdf_PDF, prev_bsdf_val)

        class EscapedRayQueue : public WorkQueue<EscapedRayWorkItem> {
        public:
            using WorkQueue::WorkQueue;

            LM_XPU_INLINE int push(RayWorkItem r) {
                EscapedRayWorkItem item{r.ray.origin(), r.ray.direction(), r.depth,
                                        r.pixel_index, r.throughput, r.prev_lsc,
                                        r.prev_bsdf_PDF, r.prev_bsdf_val};
                return WorkQueue::push(item);
            }
        };

        struct HitAreaLightWorkItem {
            HitInfo light_hit_info;
            float3 wo;
            int depth{};
            Spectrum throughput;
            LightSampleContext prev_lsc;
            float prev_bsdf_PDF{};
            Spectrum prev_bsdf_val;
            int pixel_index{};
        };

        LUMINOUS_SOA(HitAreaLightWorkItem, light_hit_info, wo, depth,
                     throughput, prev_lsc, prev_bsdf_PDF, prev_bsdf_val, pixel_index)

        using HitAreaLightQueue = WorkQueue<HitAreaLightWorkItem>;

        struct ShadowRayWorkItem {
            Ray ray;
            Spectrum Ld;
            int pixel_index{};
        };

        LUMINOUS_SOA(ShadowRayWorkItem, ray, Ld, pixel_index)

        using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;

        class Material;

        struct MaterialEvalWorkItem {
            HitInfo hit_info;
            float3 wo;
            int depth;
            int pixel_index{};
            Spectrum throughput;
        };

        LUMINOUS_SOA(MaterialEvalWorkItem, hit_info, wo, depth, pixel_index, throughput)

        using MaterialEvalQueue = WorkQueue<MaterialEvalWorkItem>;

        struct PixelSampleState {
            uint2 pixel;
            Spectrum Li;
            float3 normal;
            float3 albedo;
            float filter_weight{};
            Spectrum sensor_ray_weight;
            RaySamples ray_samples;
        };

        LUMINOUS_SOA(PixelSampleState, pixel, Li, normal, albedo, filter_weight, sensor_ray_weight, ray_samples)
    }
}