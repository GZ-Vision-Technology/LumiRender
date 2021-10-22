//
// Created by Zero on 08/09/2021.
//

#include "embree_aggregate.h"
#include "render/include/trace.h"
#include "render/scene/scene.h"
#include "process_queue.h"


namespace luminous {
    inline namespace render {

        EmbreeAggregate::EmbreeAggregate(Device *device, Context *context, const Scene *scene)
                : EmbreeAccel(device, context, scene),
                  WavefrontAggregate(scene->scene_data()) {}

        bool EmbreeAggregate::_intersect_any(Ray ray) const {
            return luminous::intersect_any((uint64_t) rtc_scene(), ray);
        }

        lstd::optional<SurfaceInteraction> EmbreeAggregate::_intersect_closest(Ray ray) const {
            PerRayData prd{_scene_data};
            bool hit = luminous::intersect_closest((uint64_t) rtc_scene(), ray, &prd);
            if (hit) {
                return {prd.compute_surface_interaction(ray)};
            } else {
                return {};
            }
        }

        void EmbreeAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                                EscapedRayQueue *escaped_ray_queue,
                                                HitAreaLightQueue *hit_area_light_queue,
                                                MaterialEvalQueue *material_eval_queue,
                                                RayQueue *next_ray_queue) const {
            parallel_for(ray_queue->size(), [=](uint task_id, uint tid) {
                RayWorkItem ray_work_item = (*ray_queue)[task_id];
                auto op_si = _intersect_closest(ray_work_item.ray);
                if (op_si) {
                    enqueue_item_after_intersect(ray_work_item, op_si.value(),
                                                 next_ray_queue, hit_area_light_queue,
                                                 material_eval_queue);
                } else {
                    enqueue_item_after_miss(ray_work_item, escaped_ray_queue);
                }
            });

        }

        void EmbreeAggregate::intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                            SOA<PixelSampleState> *pixel_sample_state) const {
            parallel_for(shadow_ray_queue->size(), [&](uint task_id, uint tid) {
                ShadowRayWorkItem item = (*shadow_ray_queue)[task_id];
                bool hit = _intersect_any(item.ray);
                record_shadow_ray_result(item, pixel_sample_state, hit);
            });
        }

        void EmbreeAggregate::intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                               SOA<PixelSampleState> *pixel_sample_state) const {

        }

    }
}