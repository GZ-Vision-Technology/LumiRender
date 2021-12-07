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
//                                           "__miss__closest",
//                                           "__miss__any"
        };

        ProgramName intersect_any_func{"__raygen__occlusion",
                                       "__closesthit__closest",
                                       "__closesthit__any",
//                                       "__miss__closest",
//                                       "__miss__any"
        };

        OptixAggregate::OptixAggregate(Device *device, Context *context, const Scene *scene)
                : OptixAccel(device, context, scene), WavefrontAggregate(scene->scene_data_host_ptr()),
                  _intersect_any(create_shader_wrapper(intersect_shader, intersect_any_func)),
                  _intersect_closet(create_shader_wrapper(intersect_shader, intersect_closest_func)) {
            auto program_groups = _intersect_closet.program_groups();
            append(program_groups, _intersect_any.program_groups());
            build_pipeline(program_groups);
            _params.reserve(1);
            _params.emplace_back();
            _params.allocate_device(1);
        }

        void OptixAggregate::intersect_closest(int max_rays, const RayQueue *ray_queue,
                                               EscapedRayQueue *escaped_ray_queue,
                                               HitAreaLightQueue *hit_area_light_queue,
                                               MaterialEvalQueue *material_eval_queue,
                                               RayQueue *next_ray_queue) const {
            _params->ray_queue = ray_queue;
            _params->next_ray_queue = next_ray_queue;
            _params->traversable_handle = OptixAccel::handle();
            _params->hit_area_light_queue = hit_area_light_queue;
            _params->escaped_ray_queue = escaped_ray_queue;
            _params->material_eval_queue = material_eval_queue;
            _params.synchronize_to_device();

            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            OPTIX_CHECK(optixLaunch(_optix_pipeline,
                                    stream,
                                    _params.device_ptr<CUdeviceptr>(),
                                    sizeof(WavefrontParams),
                                    _intersect_closet.sbt_ptr(),
                                    max_rays, 1, 1));

            _dispatcher.wait();
        }

        void OptixAggregate::intersect_any_and_compute_lighting(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                                                SOA<PixelSampleState> *pixel_sample_state) const {
            _params->shadow_ray_queue = shadow_ray_queue;
            _params->traversable_handle = OptixAccel::handle();
            _params.synchronize_to_device();

            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            OPTIX_CHECK(optixLaunch(_optix_pipeline,
                                    stream,
                                    _params.device_ptr<CUdeviceptr>(),
                                    sizeof(WavefrontParams),
                                    _intersect_any.sbt_ptr(),
                                    max_rays, 1, 1));

            _dispatcher.wait();
        }

        void OptixAggregate::intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                              SOA<PixelSampleState> *pixel_sample_state) const {

        }

    }
}