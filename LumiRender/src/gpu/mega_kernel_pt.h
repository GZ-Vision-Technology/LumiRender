//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/include/integrator.h"
#include "render/sensors/sensor.h"
#include "framework/optix_accel.h"
#include "gpu_scene.h"
#include "render/samplers/sampler_handle.h"

namespace luminous {

    inline namespace gpu {
        class MegaKernelPT : public Integrator {
        private:
            Managed<SamplerHandle> _sampler;
            Managed<SensorHandle> _camera;

        public:

//            MegaKernelPT()

            void render() override;
        };
    }
}