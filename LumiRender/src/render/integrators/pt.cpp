//
// Created by Zero on 2021/1/16.
//

#include "pt.h"
#include "render/samplers/sampler.h"
#include "render/include/trace.h"
#include "render/bxdfs/bsdf.h"
#include "render/materials/material.h"
#include "render/light_samplers/light_sampler.h"

namespace luminous {
    inline namespace render {

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler,
                             uint max_depth, float rr_threshold, bool debug) {
            PerRayData prd;
//            luminous::intersect_closest(scene_handle, ray, &prd);
//
//            if (prd.is_hit()) {
//                auto si = prd.si;
//                auto bsdf = si.op_bsdf.value();
//                auto color = bsdf.base_color();
//                return color;
//            }
//            return 0;

            Spectrum L(0.f);
            Spectrum throughput(1.f);
            SurfaceInteraction si;
            prd.sampler = &sampler;
            int bounces;
            bool found_intersection = luminous::intersect_closest(scene_handle, ray, &prd);

            for (bounces = 0; bounces < max_depth; ++bounces) {
                if (bounces == 0) {
                    if (found_intersection) {
                        si = prd.si;
                        L += throughput * si.Le(-ray.direction());
                    } else {
                        // nothing to do
                    }
                }
                BREAK_IF(!found_intersection)
                if (!si.op_bsdf) {
                    ray = si.spawn_ray(ray.direction());
                    --bounces;
                    continue;
                }

                const LightSampler *light_sampler = prd.hit_group_data()->light_sampler;
                NEEData NEE_data;
                NEE_data.debug = debug;
                Spectrum Ld = light_sampler->estimate_direct_lighting(si, sampler,
                                                                      scene_handle,
                                                                      prd.hit_group_data(), &NEE_data);
                found_intersection = NEE_data.found_intersection;
                Spectrum bsdf_ei = NEE_data.bsdf_val / NEE_data.bsdf_PDF;
                throughput *= bsdf_ei;
                L += Ld * throughput;
                ray = si.spawn_ray(NEE_data.wi);
                si = NEE_data.next_si;
                float max_comp = throughput.max_comp();
                if (max_comp < rr_threshold) {
                    float q = min(0.95f, max_comp);
                    if (q < sampler.next_1d()) {
                        break;
                    }
                    throughput /= q;
                }
            }

            return L;
        }

    } // luminous::render
} // luminous