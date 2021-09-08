//
// Created by Zero on 08/09/2021.
//

#include "embree_aggregate.h"

namespace luminous {
    inline namespace cpu {

        EmbreeAggregate::EmbreeAggregate(Device *device, Context *context, const Scene *scene)
                : EmbreeAccel(device, context, scene) {

        }

        void EmbreeAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                           EscapedRayQueue *escaped_ray_queue,
                                           HitAreaLightQueue *hit_area_light_queue,
                                           MaterialEvalQueue *material_eval_queue,
                                           RayQueue *next_ray_queue) {

        }

        void EmbreeAggregate::intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                            SOA<PixelSampleState> *pixel_sample_state) {

        }

        void EmbreeAggregate::intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                               SOA<PixelSampleState> *pixel_sample_state) {

        }
    }
}