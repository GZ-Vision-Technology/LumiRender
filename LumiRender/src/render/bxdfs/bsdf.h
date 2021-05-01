//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/include/interaction.h"
#include "render/bxdfs/bxdf.h"

namespace luminous {
    inline namespace render {

        class BSDF {
        private:
            BxDF _bxdf;
            float3 _ng;
            Frame _shading_frame;
        public:
            XPU BSDF() = default;

            NDSC_XPU bool is_non_specular() const {
                return luminous::is_non_specular(_bxdf.flags());
            }

            NDSC_XPU bool is_reflective() const {
                return luminous::is_reflective(_bxdf.flags());
            }

            NDSC_XPU bool is_transmissive() const {
                return luminous::is_transmissive(_bxdf.flags());
            }

            NDSC_XPU bool is_diffuse() const {
                return luminous::is_diffuse(_bxdf.flags());
            }

            NDSC_XPU bool is_glossy() const {
                return luminous::is_glossy(_bxdf.flags());
            }

            NDSC_XPU bool is_specular() const {
                return luminous::is_specular(_bxdf.flags());
            }
        };

    } // luminous::render
} // luminous