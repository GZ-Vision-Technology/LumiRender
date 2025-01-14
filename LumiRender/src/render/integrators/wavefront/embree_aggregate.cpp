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
                  WavefrontAggregate(scene->scene_data_host_ptr()) {}

        bool EmbreeAggregate::_intersect_any(Ray ray) const {
            return luminous::intersect_any((uint64_t) rtc_scene(), ray);
        }

        HitContext EmbreeAggregate::_intersect_closest(Ray ray) const {
            HitContext hit_ctx{_scene_data};
            luminous::intersect_closest((uint64_t) rtc_scene(), ray, &hit_ctx.hit_info);
            return hit_ctx;
        }

        void EmbreeAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                                EscapedRayQueue *escaped_ray_queue,
                                                HitAreaLightQueue *hit_area_light_queue,
                                                MaterialEvalQueue *material_eval_queue,
                                                RayQueue *next_ray_queue) const {
            parallel_for(ray_queue->size(), [=](uint task_id, uint tid) {
                RayWorkItem ray_work_item = (*ray_queue)[task_id];
                auto hit_ctx = _intersect_closest(ray_work_item.ray);
                if (hit_ctx.is_hit()) {
                    enqueue_item_after_intersect(ray_work_item, hit_ctx,
                                                 next_ray_queue, hit_area_light_queue,
                                                 material_eval_queue);
                } else {
                    enqueue_item_after_miss(ray_work_item, escaped_ray_queue);
                }
            });

        }

        void EmbreeAggregate::intersect_any_and_compute_lighting(int max_rays, ShadowRayQueue *shadow_ray_queue,
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