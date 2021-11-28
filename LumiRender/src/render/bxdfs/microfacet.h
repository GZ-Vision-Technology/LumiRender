//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {
        enum MDType {
            GGX,
            Beckmann,
        };

        class MicrofacetDistribution {
        private:
            float _alpha_x{};
            float _alpha_y{};
            MDType _md_type{};
        public:
            LM_XPU MicrofacetDistribution(float alpha_x, float alpha_y, MDType md_type = GGX)
                    : _alpha_x(alpha_x),
                      _alpha_y(alpha_y),
                      _md_type(md_type) {

            }

            LM_ND_XPU float D(const float3 &wh) const;

            /**
             * ¦«(¦Ø) = A-(¦Ø) / (A+(¦Ø) - A-(¦Ø))
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float lambda(const float3 &w) const;

            /**
             * smith occlusion function
             * G1(¦Ø) = 1 / (¦«(¦Ø) + 1)
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float G1(const float3 &w) const {
                return 1 / (1 + lambda(w));
            }

            /**
             * G(¦Øo, ¦Øi) = 1 / (¦«(¦Øo) + ¦«(¦Øi) + 1)
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float G(const float3 &wo, const float3 &wi) const {
                return 1 / (1 + lambda(wo) + lambda(wi));
            }

            LM_ND_XPU float3 sample_wh(const float3 &wo, const float2 &u) const;

            /**
             * @param  wo
             * @param  wh :normal of microfacet
             * @return
             */
            LM_ND_XPU float PDF_dir(const float3 &wo, const float3 &wh) const;

        };
    }
}