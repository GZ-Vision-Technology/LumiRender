//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "gpu/integrators/integrator.h"
#include "render/sensors/camera_base.h"
#include "gpu/accel/megakernel_optix_accel.h"
#include "gpu/gpu_scene.h"
#include "render/samplers/sampler.h"
#include "work_items.h"

namespace luminous {
    inline namespace gpu {
        class WavefrontPT : public Integrator {
        private:
            RayQueue *_ray_queue[2];
            int _scanline_per_pass{};
            int _max_queue_size{};
        public:
            void render() override;


        };
    }
}