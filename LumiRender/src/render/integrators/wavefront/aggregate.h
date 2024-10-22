//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "core/concepts.h"
#include "work_queue.h"
#include "work_items.h"

namespace luminous {
    inline namespace render {

        class SceneData;

        class WavefrontAggregate : public Noncopyable {
        protected:
            const SceneData *_scene_data;
        public:
            explicit WavefrontAggregate(const SceneData *scene_data)
                    : _scene_data(scene_data) {}

            virtual void intersect_closest(int max_rays, const RayQueue *ray_queue,
                                           EscapedRayQueue *escaped_ray_queue,
                                           HitAreaLightQueue *hit_area_light_queue,
                                           MaterialEvalQueue *material_eval_queue,
                                           RayQueue *next_ray_queue) const = 0;

            virtual void intersect_any_and_compute_lighting(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                                            SOA<PixelSampleState> *pixel_sample_state) const = 0;

            virtual void intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                          SOA<PixelSampleState> *pixel_sample_state) const = 0;

            virtual ~WavefrontAggregate() = default;
        };
    }
}