//
// Created by Zero on 25/08/2021.
//

#include "optix_aggregate.h"

namespace luminous {
    inline namespace gpu {

        OptixAggregate::OptixAggregate(Device *device, Context *context, const Scene *scene)
                : OptixAccel(device, context, scene) {

        }

        void OptixAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                               EscapedRayQueue *escaped_ray_queue,
                                               HitAreaLightQueue *hit_area_light_queue,
                                               MaterialEvalQueue *material_eval_queue,
                                               RayQueue *next_ray_queue) {

        }

        void OptixAggregate::intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                           SOA<PixelSampleState> *pixel_sample_state) {

        }

        void OptixAggregate::intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                              SOA<PixelSampleState> *pixel_sample_state) {

        }

    }
}