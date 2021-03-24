//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "render/include/integrator.h"
#include "render/sensors/sensor.h"
#include "framework/optix_accel.h"
#include "gpu_scene.h"
#include "render/samplers/sampler_handle.h"

namespace luminous {
    inline namespace gpu {
        class WavefrontPT : public Integrator {
        private:

        public:
            void render() override;
        };
    }
}