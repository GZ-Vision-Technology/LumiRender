//
// Created by Zero on 2021/1/16.
//

#include "pt.h"
#include "render/samplers/sampler.h"
#include "render/include/trace.h"
#include "render/bxdfs/bsdf.h"
#include "render/materials/material.h"

namespace luminous {
    inline namespace render {

        NDSC_XPU Spectrum Li(Ray ray, uint64_t scene_handle, Sampler &sampler ,uint max_depth, float rr_threshold) {
            PerRayData prd;
            luminous::intersect_closest(scene_handle, ray, &prd);

            if (prd.is_hit()) {
                auto si = prd.get_surface_interaction();
                const Material *mat = si.material;
                auto bsdf = mat->get_BSDF(si, prd.hit_group_data);
                auto color = bsdf.base_color();
                return color;
            }


            return  0;
        }

    } // luminous::render
} // luminous