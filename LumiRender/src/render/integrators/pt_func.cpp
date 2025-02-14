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

        LM_ND_XPU PixelInfo path_tracing(Ray ray, uint64_t scene_handle, Sampler &sampler, uint min_depth,
                                         uint max_depth, float rr_threshold, const SceneData *scene_data, bool debug) {
            HitContext hit_ctx{scene_data};

            Spectrum L(0.f);
            Spectrum throughput(1.f);
            SurfaceInteraction si;
            const LightSampler *light_sampler = hit_ctx.scene_data()->light_sampler;
            int bounces;
            bool found_intersection = luminous::intersect_closest(scene_handle, ray, &hit_ctx.hit_info);

            PixelInfo pixel_info;

            float eta_scale = 1.f;

            bool fill_denoise_data = false;

            if (found_intersection) {
                si = hit_ctx.compute_surface_interaction(ray);
                L += throughput * si.Le(-ray.direction(), scene_data);
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
                if (lm_unlikely(!si.has_material())) {
                    ray = si.spawn_ray(ray.direction());
                    found_intersection = luminous::intersect_closest(scene_handle, ray, &hit_ctx.hit_info);
                    si = hit_ctx.compute_surface_interaction(ray);
                    --bounces;
                    continue;
                }
                PathVertex vertex;
                vertex.debug = debug;

                auto bsdf = si.compute_BSDF(scene_data);
                Spectrum mis_light = light_sampler->MIS_sample_light(si, bsdf, sampler, scene_handle, scene_data, debug);
                Spectrum mis_bsdf = light_sampler->MIS_sample_BSDF(si, bsdf, sampler, scene_handle, &vertex, scene_data);

                found_intersection = vertex.found_intersection;
                Spectrum bsdf_ei = vertex.bsdf_val / vertex.bsdf_PDF;

                Spectrum Ld = mis_light + mis_bsdf;

                L += Ld * throughput;
                throughput *= bsdf_ei;

                DCHECK(!has_invalid(L));

                if (is_transmissive(vertex.bxdf_flags)) {
                    eta_scale *= dot(si.wo, si.g_uvn.normal()) > 0 ? sqr(vertex.eta) : sqr(rcp(vertex.eta));
                }

                if (!fill_denoise_data && is_non_specular(vertex.bxdf_flags)) {
                    pixel_info.normal = si.s_uvn.normal();
                    pixel_info.albedo = vertex.albedo;
                    fill_denoise_data = true;
                }

                float rr = sampler.next_1d();
                Spectrum rr_throughput = throughput * eta_scale;
                float max_comp = rr_throughput.max_comp();
                if (max_comp < rr_threshold && bounces >= min_depth) {
                    float q = min(0.95f, max_comp);
                    if (q < rr) {
                        break;
                    }
                    throughput /= q;
                }
                si = vertex.next_si;
            }

            pixel_info.Li = L;

            return pixel_info;
        }

    } // luminous::render
} // luminous