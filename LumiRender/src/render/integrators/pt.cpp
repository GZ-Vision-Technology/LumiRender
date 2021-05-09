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

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler, uint max_depth, float rr_threshold) {
            PerRayData prd;
            luminous::intersect_closest(scene_handle, ray, &prd);

            if (prd.is_hit()) {
                auto si = prd.get_surface_interaction();
                const Material *mat = si.material;
                auto bsdf = mat->get_BSDF(si, prd.hit_group_data);
                auto color = bsdf.base_color();
                return color;
            }
//            return 0;

            Spectrum L(0.f);
            Spectrum throughput(1.f);
            int bounces;
            bool found_intersection;
            SurfaceInteraction si;
            for (bounces = 0;; ++bounces) {
                found_intersection = luminous::intersect_closest(scene_handle, ray, &prd);
                if (bounces == 0) {
                    if (found_intersection) {
                        si = prd.get_surface_interaction();
                        L += throughput * si.Le(-ray.direction());
                    } else {
                        // nothing to do
                    }
                }

                BREAK_IF(!found_intersection || bounces >= max_depth)

                const LightSampler *light_sampler = prd.hit_group_data->light_sampler;

                auto op_bsdf = si.get_BSDF(prd.hit_group_data);
                if (!op_bsdf) {
                    ray = si.spawn_ray(ray.direction());
                    --bounces;
                    continue;
                }

                NEEData NEE_data;
                Spectrum Ld = light_sampler->estimate_direct_lighting(si, op_bsdf.value(), sampler,
                                                                      scene_handle,
                                                                      prd.hit_group_data, &NEE_data);
                found_intersection = NEE_data.found_intersection;
            }


            return 0;
        }

    } // luminous::render
} // luminous