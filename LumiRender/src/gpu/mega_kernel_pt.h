//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/sensors/sensor.h"
#include "framework/optix_accel.h"
#include "scene.h"
#include "render/samplers/sampler_handle.h"

namespace luminous {



    inline namespace gpu {
        class MegaKernelPT {
        private:
            SensorHandle _camera;
            Buffer<SensorHandle> _d_camera{nullptr};

            SamplerHandle _sampler;
            Buffer<SamplerHandle> _d_sampler{nullptr};

            LaunchParams _launch_params{};
            Buffer<LaunchParams> _d_launch_params{nullptr};
        };
    }
}