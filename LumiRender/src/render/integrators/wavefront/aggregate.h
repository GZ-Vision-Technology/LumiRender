//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "core/concepts.h"
#include "work_queue.h"
#include "work_items.h"

namespace luminous {
    inline namespace render {
        class WavefrontAggregate : public Noncopyable {
        public:
            virtual void intersect_closest(int max_rays, const RayQueue *ray_queue,
                                           EscapedRayQueue *escaped_ray_queue,
                                           HitAreaLightQueue *hit_area_light_queue,
                                           MaterialEvalQueue *material_eval_queue,
                                           RayQueue *next_ray_queue) = 0;

            virtual void intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                       SOA<PixelSampleState> *pixel_sample_state) = 0;

            virtual void intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                          SOA<PixelSampleState> *pixel_sample_state) = 0;

            virtual ~WavefrontAggregate() {}
        };
    }
}