//
// Created by Zero on 25/08/2021.
//


#pragma once

#include "gpu/accel/optix_accel.h"
#include "render/integrators/wavefront/aggregate.h"
#include "render/integrators/wavefront/params.h"

namespace luminous {
    inline namespace gpu {
        class OptixAggregate : public OptixAccel, public WavefrontAggregate {
        private:
            ShaderWrapper _intersect_closet;
            ShaderWrapper _intersect_any;
            mutable Managed<WavefrontParams, WavefrontParams> _params{_device};
        public:
            OptixAggregate(Device *device, Context *context, const Scene *scene);

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