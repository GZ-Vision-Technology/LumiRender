//
// Created by Zero on 2021/3/17.
//


#pragma once

#include <optix.h>
#include "base_libs/math/common.h"
#include "render/sensors/sensor.h"
#include "render/samplers/sampler.h"
#include "render/scene/scene_data.h"
#include "optix_defines.h"

namespace luminous {
    inline namespace gpu {
        struct LaunchParams {
            OptixTraversableHandle traversable_handle;
            uint frame_index;
            uint max_depth;
            float rr_threshold;
            Sensor *camera;
            const Sampler *sampler;
        };
    }
}