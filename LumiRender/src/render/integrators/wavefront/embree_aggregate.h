//
// Created by Zero on 08/09/2021.
//


#pragma once

#include "render/integrators/wavefront/aggregate.h"
#include "cpu/accel/embree_accel.h"

namespace luminous {
    inline namespace render {
        class EmbreeAggregate : public EmbreeAccel, public WavefrontAggregate {

        private:
            LM_NODISCARD bool _intersect_any(Ray ray) const;

            LM_NODISCARD HitContext _intersect_closest(Ray ray) const;
        public:
            EmbreeAggregate(Device *device, Context *context, const Scene *scene);

            void intersect_closest(int max_rays, const RayQueue *ray_queue,
                                   EscapedRayQueue *escaped_ray_queue,
                                   HitAreaLightQueue *hit_area_light_queue,
                                   MaterialEvalQueue *material_eval_queue,
                                   RayQueue *next_ray_queue) const override;

            void intersect_any_and_record_direct_lighting(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                                          SOA<PixelSampleState> *pixel_sample_state) const override;

            void intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                  SOA<PixelSampleState> *pixel_sample_state) const override;
        };
    }
}