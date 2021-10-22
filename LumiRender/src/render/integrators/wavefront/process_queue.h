//
// Created by Zero on 22/10/2021.
//


#pragma once

#include "work_items.h"

namespace luminous {
    inline namespace render {
        LM_XPU void enqueue_item_after_miss(RayWorkItem r, EscapedRayQueue *escaped_ray_queue);

        LM_XPU void record_shadow_ray_result(ShadowRayWorkItem w,
                                             SOA<PixelSampleState> *pixel_sample_state,
                                             bool found_intersection);

        LM_XPU void enqueue_item_after_intersect(RayWorkItem r, SurfaceInteraction si,
                                                 RayQueue *next_ray_queue,
                                                 HitAreaLightQueue *hit_area_light_queue,
                                                 MaterialEvalQueue *material_eval_queue);
    }
}