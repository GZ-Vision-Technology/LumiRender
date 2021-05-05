//
// Created by Zero on 2021/5/5.
//


#pragma once

#include "graphics/math/common.h"
#include "interaction.h"
#include "gpu/shaders/optix_kernels.h"

namespace luminous {
    inline namespace render {


        XPU_INLINE bool ray_intersect(uint64_t traversable_handle, Ray ray, RadiancePRD *prd) {
#ifdef IS_GPU_CODE
            return traceRadiance((OptixTraversableHandle)traversable_handle, ray, prd);
#else
            // CPU is not implemented
            assert(0);
            return false;
#endif
        }

        XPU_INLINE bool ray_occluded(uint64_t traversable_handle, Ray ray) {
#ifdef IS_GPU_CODE
            return traceOcclusion((OptixTraversableHandle)traversable_handle, ray);
#else
            // CPU is not implemented
            assert(0);
            return false;
#endif
        }

    } // luminous::render
} // luminous
