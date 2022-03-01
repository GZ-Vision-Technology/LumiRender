//
// Created by Zero on 03/09/2021.
//


#pragma once


#include "gpu/accel/optix_defines.h"
#include "render/integrators/wavefront/work_items.h"

namespace luminous {
    inline namespace render {
        struct WavefrontParams {
        public:
            const RayQueue *ray_queue;
            RayQueue *next_ray_queue;
            uint64_t traversable_handle;
            HitAreaLightQueue *hit_area_light_queue;
            EscapedRayQueue *escaped_ray_queue;
            MaterialEvalQueue *material_eval_queue;
            ShadowRayQueue *shadow_ray_queue;
            const SceneData *scene_data;
            SOA <PixelSampleState> *pixel_sample_state;
        public:
            WavefrontParams() = default;
        };
    }
}