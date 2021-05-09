//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "graphics/geometry/common.h"
#include "graphics/optics/common.h"

namespace luminous {
    inline namespace render {

        class Sampler;

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler, uint max_depth, float rr_threshold);

    }
}