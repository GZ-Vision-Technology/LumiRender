//
// Created by Zero on 2021/1/16.
//

#include "pt.h"
#include "render/samplers/sampler.h"
#include "render/include/trace.h"

namespace luminous {
    inline namespace render {

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler ,uint max_depth, float rr_threshold) {
            RadiancePRD prd;
            luminous::intersect_closest(scene_handle, ray, &prd);
            return prd.is_hit() ? 1 : 0;
        }

    } // luminous::render
} // luminous