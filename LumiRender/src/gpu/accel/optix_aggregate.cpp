//
// Created by Zero on 25/08/2021.
//

#include "optix_aggregate.h"
#include <iosfwd>

extern "C" char intersect_shader[];

namespace luminous {
    inline namespace gpu {

        ProgramName intersect_closest_func{"__raygen__find_closest",
                                           "__closesthit__closest",
                                           "__closesthit__any",
                                           "__miss__closest",
                                           "__miss__any"};

        ProgramName intersect_any_func{"__raygen__occlusion",
                                       "__closesthit__closest",
                                       "__closesthit__any",
                                       "__miss__closest",
                                       "__miss__any"};

        OptixAggregate::OptixAggregate(Device *device, Context *context, const Scene *scene)
                : OptixAccel(device, context, scene)
//                _intersect_any(create_shader_wrapper(intersect_shader, intersect_any_func))
//                _intersect_closet(create_shader_wrapper(intersect_shader, intersect_closest_func))
                {

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