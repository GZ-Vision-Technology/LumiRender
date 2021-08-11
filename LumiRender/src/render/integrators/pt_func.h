//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "base_libs/geometry/common.h"
#include "base_libs/optics/common.h"

namespace luminous {
    inline namespace render {

        class Sampler;

        class SceneData;

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler,
                             uint max_depth, float rr_threshold, bool debug = false,
                             const SceneData *scene_data = nullptr);

    }
}