//
// Created by Zero on 30/08/2021.
//


#pragma once

#include "render/integrators/integrator.h"
#include "gpu/framework/cuda_impl.h"
#include "gpu/accel/optix_accel.h"

namespace luminous {
    inline namespace gpu {
        class GPUIntegrator : public Integrator {
        protected:
            std::unique_ptr<OptixAccel> _accel;

        public:
            GPUIntegrator(const SP <Device> &device, Context *context)
                    : Integrator(device, context) {}
        };
    }
}