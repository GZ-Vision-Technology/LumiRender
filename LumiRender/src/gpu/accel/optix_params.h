//
// Created by Zero on 2021/3/17.
//


#pragma once

#include <optix.h>
#include "base_libs/math/common.h"
#include "render/samplers/sampler.h"
#include "optix_defines.h"

namespace luminous {

    inline namespace render {
        class Sensor;
        class SceneData;
    }

    inline namespace gpu {
        struct LaunchParams {
            OptixTraversableHandle traversable_handle;
            uint frame_index;
            uint max_depth;
            float rr_threshold;
            Sensor *camera;
            const SceneData *scene_data;
            const Sampler *sampler;
        };
    }
}