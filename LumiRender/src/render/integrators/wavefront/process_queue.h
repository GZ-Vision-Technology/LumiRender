//
// Created by Zero on 22/10/2021.
//


#pragma once

#include "work_items.h"

namespace luminous {
    inline namespace render {
        LM_XPU void enqueue_item_after_miss(RayWorkItem r, EscapedRayQueue *escaped_ray_queue);

        LM_XPU void RecordShadowRayResult(ShadowRayWorkItem w,
                                          SOA<PixelSampleState> *pixelSampleState,
                                          bool foundIntersection);

        LM_XPU void enqueue_item_after_intersect(RayWorkItem r, float tMax, SurfaceInteraction si,
                                                 RayQueue *next_ray_queue,
                                                 HitAreaLightQueue *hit_area_light_queue,
                                                 MaterialEvalQueue *material_eval_queue);
    }
}