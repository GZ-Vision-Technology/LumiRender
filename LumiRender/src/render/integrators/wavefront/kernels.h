//
// Created by Zero on 14/10/2021.
//


#pragma once

#include <render/sensors/shader_include.h>
#include "base_libs/math/common.h"
#include "work_items.h"

namespace luminous {
    inline namespace render {

        struct RTParam {
            Sampler *sampler;
            Sensor *camera;
            int frame_index;

            RTParam() = default;

            RTParam(Sampler *sampler, Sensor *camera, int frame_index)
                    : sampler(sampler), camera(camera), frame_index(frame_index) {}
        };

        CPU_ONLY(void set_rt_param(RTParam *param);)

        LM_XPU void generate_primary_ray(int task_id, int n_item,int y0, RayQueue *ray_queue, const Sampler *_sampler,
                                         SOA<PixelSampleState> *pixel_sample_state);

        LM_XPU void reset_ray_queue(int task_id, int n_item, RayQueue *ray_queue);

        LM_XPU void reset_queues(int task_id, int n_item, RayQueue *ray_queue,
                                 HitAreaLightQueue *hit_area_light_queue,
                                 ShadowRayQueue *shadow_ray_queue,
                                 EscapedRayQueue *escaped_ray_queue,
                                 MaterialEvalQueue *material_eval_queue);

        LM_XPU void generate_ray_samples(int task_id, int n_item, const RayQueue *ray_queue,
                                         SOA<PixelSampleState> *pixel_sample_state);

        LM_XPU void process_escape_ray(int task_id, int n_item,
                                       EscapedRayQueue *escaped_ray_queue);

        LM_XPU void process_emission(int task_id, int n_item,
                                     HitAreaLightQueue *hit_area_light_queue);

        LM_XPU void eval_BSDFs(int task_id, int n_item,
                               MaterialEvalQueue *material_eval_queue);
    }
}