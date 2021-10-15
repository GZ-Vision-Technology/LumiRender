//
// Created by Zero on 14/10/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "work_items.h"

namespace luminous {
    inline namespace render {
        LM_XPU void generate_primary_ray(int n_item, RayQueue *ray_queue, const Sampler *sampler,
                                         SOA<PixelSampleState> *pixel_sample_state);
    }
}