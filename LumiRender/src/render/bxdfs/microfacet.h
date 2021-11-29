//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {
        enum MicrofacetType {
            GGX,
            Beckmann,
        };

        class MicrofacetDistribution {
        private:
            float _alpha_x{};
            float _alpha_y{};
            MicrofacetType _type{};
        public:
            LM_XPU MicrofacetDistribution(float alpha_x, float alpha_y, MicrofacetType md_type = GGX)
                    : _alpha_x(alpha_x),
                      _alpha_y(alpha_y),
                      _type(md_type) {

            }

            static float roughness_to_alpha(float roughness) {
                roughness = std::max(roughness, (float) 1e-3);
                float x = std::log(roughness);
                return 1.62142f +
                       0.819955f * x +
                       0.1734f * Pow<2>(x) +
                       0.0171201f * Pow<3>(x) +
                       0.000640711f * Pow<4>(x);
            }

            /**
             *  beckmann
             *
             *             e^[-(tan_theta_h)^2 ((cos_theta_h)^2/ax^2 + (sin_theta_h)^2/ay^2)]
             * D(wh) = -------------------------------------------------------------------------
             *                                PI ax ay (cos_theta_h)^4
             *
             *  GGX
             *                                                    1
             * D(wh) = ---------------------------------------------------------------------------------------------------
             *             PI ax ay (cos_theta_h)^4 [1 + (tan_theta_h)^2 ((cos_theta_h)^2/ax^2 + (sin_theta_h)^2/ay^2)]^2
             *
             * from http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html
             *
             * @param wh
             * @return
             */
            LM_ND_XPU float D(const float3 &wh) const;

            /**
             * lambda(w) = A-(w) / (A+(w) - A-(w))
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float lambda(const float3 &w) const;

            /**
             * smith occlusion function
             * G1(w) = 1 / (lambda(w) + 1)
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float G1(const float3 &w) const {
                return 1 / (1 + lambda(w));
            }

            /**
             * G(wo, wi) = 1 / (lambda(wo) + lambda(wi) + 1)
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