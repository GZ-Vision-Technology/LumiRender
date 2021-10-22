//
// Created by Zero on 22/10/2021.
//

#include "process_queue.h"

namespace luminous {
    inline namespace render {
        void enqueue_item_after_miss(RayWorkItem r, EscapedRayQueue *escaped_ray_queue) {

        }

        void RecordShadowRayResult(ShadowRayWorkItem w,
                                   SOA<PixelSampleState> *pixelSampleState,
                                   bool foundIntersection) {

        }

        void enqueue_item_after_intersect(RayWorkItem r, float tMax, SurfaceInteraction si,
                                          RayQueue *next_ray_queue,
                                          HitAreaLightQueue *hit_area_light_queue,
                                          MaterialEvalQueue *material_eval_queue) {

        }
    }
}