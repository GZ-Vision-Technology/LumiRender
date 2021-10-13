//
// Created by Zero on 25/08/2021.
//


#pragma once

#include "optix_accel.h"
#include "render/integrators/wavefront/aggregate.h"
#include "render/integrators/wavefront/params.h"

namespace luminous {
    inline namespace gpu {
        class OptixAggregate : public OptixAccel, public WavefrontAggregate {
        private:

        public:
            OptixAggregate(Device *device, Context *context, const Scene *scene);

            void intersect_closest(int max_rays, const RayQueue *ray_queue,
                                           EscapedRayQueue *escaped_ray_queue,
                                           HitAreaLightQueue *hit_area_light_queue,
                                           MaterialEvalQueue *material_eval_queue,
                                           RayQueue *next_ray_queue) override;

            void intersect_any(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                       SOA<PixelSampleState> *pixel_sample_state) override;

            void intersect_any_tr(int max_rays, ShadowRayQueue *shadow_ray_queue,
                                          SOA<PixelSampleState> *pixel_sample_state) override;
        };
    }
}