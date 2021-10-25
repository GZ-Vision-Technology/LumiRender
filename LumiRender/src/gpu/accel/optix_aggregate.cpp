//
// Created by Zero on 25/08/2021.
//

#include "optix_aggregate.h"
#include "render/scene/scene.h"


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
                : OptixAccel(device, context, scene), WavefrontAggregate(scene->scene_data()),
                  _intersect_any(create_shader_wrapper(intersect_shader, intersect_any_func)),
                  _intersect_closet(create_shader_wrapper(intersect_shader, intersect_closest_func)) {
            auto program_groups = _intersect_closet.program_groups();
            append(program_groups, _intersect_any.program_groups());
            build_pipeline(program_groups);
            _params.emplace_back();
        }

        void OptixAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                               EscapedRayQueue *escaped_ray_queue,
                                               HitAreaLightQueue *hit_area_light_queue,
                                               MaterialEvalQueue *material_eval_queue,
                                               RayQueue *next_ray_queue) const {
//            _params[0].ray_queue = ray_queue;
//            params.next_ray_queue = next_ray_queue;
//            params.traversable_handle = OptixAccel::handle();
//            params.hit_area_light_queue = hit_area_light_queue;
//            params.escaped_ray_queue = escaped_ray_queue;
//            params.material_eval_queue = material_eval_queue;

        }

        void OptixAggregate::intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                           SOA<PixelSampleState> *pixel_sample_state) const {

        }

        void OptixAggregate::intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                              SOA<PixelSampleState> *pixel_sample_state) const {

        }

    }
}