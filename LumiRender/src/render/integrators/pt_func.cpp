//
// Created by Zero on 2021/1/16.
//

#include "pt_func.h"
#include "render/samplers/sampler.h"
#include "render/include/trace.h"
#include "render/materials/material.h"
#include "render/light_samplers/common.h"
#include "render/light_samplers/light_sampler.h"
#include "render/scene/scene_data.h"
#include "render/lights/light.h"

namespace luminous {
    inline namespace render {

        LM_ND_XPU PixelInfo path_tracing(Ray ray, uint64_t scene_handle, Sampler &sampler,
                                         uint max_depth, float rr_threshold, bool debug,
                                         const SceneData *scene_data) {
            HitContext hit_ctx{scene_data};

            Spectrum L(0.f);
            Spectrum throughput(1.f);
            SurfaceInteraction si;
            const LightSampler *light_sampler = hit_ctx.scene_data()->light_sampler;
            int bounces;
            bool found_intersection = luminous::intersect_closest(scene_handle, ray, &hit_ctx.hit_info);

            PixelInfo pixel_info;

            if (found_intersection) {
                si = hit_ctx.compute_surface_interaction(ray);
                L += throughput * si.Le(-ray.direction(), scene_data);
                pixel_info.normal = si.s_uvn.normal;
                if (lm_likely(si.op_bsdf)) {
                    pixel_info.albedo = make_float3(si.op_bsdf->base_color());
                }
            } else {
                Spectrum env_color = light_sampler->on_miss(ray.direction(), hit_ctx.scene_data(),
                                                                                  throughput);
                L += env_color;
                pixel_info.albedo = env_color.vec();
                pixel_info.Li = L;
                return pixel_info;
            }
            for (bounces = 0; bounces < max_depth; ++bounces) {
                BREAK_IF(!found_intersection)
                if (lm_unlikely(!si.op_bsdf)) {
                    ray = si.spawn_ray(ray.direction());
                    found_intersection = luminous::intersect_closest(scene_handle, ray, &hit_ctx.hit_info);
                    si = hit_ctx.compute_surface_interaction(ray);
                    --bounces;
                    continue;
                }
                NEEData NEE_data;
                NEE_data.debug = debug;
                Spectrum Ld = light_sampler->estimate_direct_lighting(si, sampler,
                                                                      scene_handle,
                                                                      hit_ctx.scene_data(), &NEE_data);
                found_intersection = NEE_data.found_intersection;
                Spectrum bsdf_ei = NEE_data.bsdf_val / NEE_data.bsdf_PDF;

                throughput *= bsdf_ei;
                L += Ld * throughput;
                float max_comp = throughput.max_comp();
                if (max_comp < rr_threshold) {
                    float q = min(0.95f, max_comp);
                    if (q < sampler.next_1d()) {
                        break;
                    }
                    throughput /= q;
                }
                si = NEE_data.next_si;
            }

            pixel_info.Li = L;

            return pixel_info;
        }

    } // luminous::render
} // luminous