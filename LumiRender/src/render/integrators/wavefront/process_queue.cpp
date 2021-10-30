//
// Created by Zero on 22/10/2021.
//

#include "process_queue.h"
#include "render/scene/shader_include.h"
#include "render/lights/shader_include.h"
#include "render/materials/shader_include.h"

namespace luminous {
    inline namespace render {
        void enqueue_item_after_miss(RayWorkItem r, EscapedRayQueue *escaped_ray_queue) {
            // todo process medium
            escaped_ray_queue->push(r);
        }

        void record_shadow_ray_result(ShadowRayWorkItem w,
                                      SOA<PixelSampleState> *pixel_sample_state,
                                      bool found_intersection) {
            if (found_intersection) {
                return;
            }
            Spectrum L = w.Ld;
            Spectrum L_pixel = pixel_sample_state->Li[w.pixel_index];
            pixel_sample_state->Li[w.pixel_index] = L_pixel + L;
        }

        void enqueue_item_after_intersect(RayWorkItem r, PerRayData prd,
                                          RayQueue *next_ray_queue,
                                          HitAreaLightQueue *hit_area_light_queue,
                                          MaterialEvalQueue *material_eval_queue) {
            // todo process medium

            auto si = prd.compute_surface_interaction(r.ray);

            if (!si.has_material()) {
                Ray new_ray = si.spawn_ray(r.ray.direction());
                next_ray_queue->push_secondary_ray(new_ray, r.depth, r.prev_lsc,
                                                   r.throughput, r.eta_scale,
                                                   r.specular_bounce,
                                                   r.any_non_specular_bounces,
                                                   r.pixel_index);
                return;
            }

            if (si.has_emission()) {
                auto scene_data = prd.scene_data();
                float PMF = scene_data->compute_prim_PMF(prd.hit_point);
                si.update_PDF_pos(PMF);
                HitAreaLightWorkItem item{si.light, AreaLightEvalContext{si}, si.wo, r.depth, r.throughput, r.prev_lsc,
                                          r.specular_bounce, r.pixel_index};
                hit_area_light_queue->push(item);

            }

            MaterialEvalWorkItem item{si.pos, si.g_uvn.normal, si.s_uvn.normal,
                                      si.uv, si.wo, r.any_non_specular_bounces,
                                      r.pixel_index, r.throughput};
            material_eval_queue->push(item);
        }
    }
}