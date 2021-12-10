//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "base.h"

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
            constexpr static MicrofacetType _type{GGX};
        public:
            LM_XPU MicrofacetDistribution() = default;

            LM_XPU MicrofacetDistribution(float alpha_x, float alpha_y)
                    : _alpha_x(alpha_x),
                      _alpha_y(alpha_y) {

            }

            LM_ND_XPU static float roughness_to_alpha(float roughness) {
                roughness = std::max(roughness, (float) 1e-3);
                float x = std::log(roughness);
                return 1.62142f +
                       0.819955f * x +
                       0.1734f * Pow<2>(x) +
                       0.0171201f * Pow<3>(x) +
                       0.000640711f * Pow<4>(x);
            }

            LM_ND_XPU bool effectively_smooth() const {
                return std::max(_alpha_x, _alpha_y) < 1e-3f;
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
                auto ret = 1 / (1 + lambda(wo) + lambda(wi));
                return ret;
            }

            LM_ND_XPU float3 sample_wh(const float3 &wo, const float2 &u) const;

            /**
             * @param  wo
             * @param  wh :normal of microfacet
             * @return
             */
            LM_ND_XPU float PDF_wh(const float3 &wo, const float3 &wh) const;

            /**
             * pwi(wi) = dwh / dwi * pwh(wh) = pwh(wh) / 4cos_theta_h
             * @param PDF_wh
             * @param wo
             * @param wh
             * @return
             */
            LM_ND_XPU float PDF_wi_reflection(float PDF_wh, float3 wo, float3 wh) const {
                float ret = PDF_wh / (4 * abs_dot(wo, wh));
                DCHECK(!is_invalid(ret));
                return ret;
            }

            LM_ND_XPU float PDF_wi_reflection(float3 wo, float3 wh) const {
                return PDF_wi_reflection(PDF_wh(wo, wh), wo, wh);
            }

            /**
             * dwh  dwi
             *                   eta_i^2 |wi dot wh|
             * dwh/dwi = -----------------------------------------
             *            [eta_o(wh dot wo) + eta_i(wi dot wh)]^2
             * @tparam T
             * @param PDF_wh
             * @param eta eta_i / eta_o
             * @return
             */
            LM_ND_XPU float PDF_wi_transmission(float PDF_wh, float3 wo, float3 wh, float3 wi, float eta) const {
                float denom = sqr(dot(wi, wh) * eta + dot(wo, wh));
                float dwh_dwi = abs_dot(wi, wh) / denom;
                float ret = PDF_wh * dwh_dwi;
                DCHECK(!is_invalid(ret));
                return ret;
            }

            LM_ND_XPU float PDF_wi_transmission(float3 wo, float3 wh, float3 wi, float eta) const {
                return PDF_wi_transmission(PDF_wh(wh, wo), wo, wh, wi, eta);
            }

            template<typename T>
            LM_ND_XPU T BRDF(float3 wo, float3 wh, float3 wi, T Fr,
                             float cos_theta_i, float cos_theta_o,
                             TransportMode mode = TransportMode::Radiance) const {
                auto ret = D(wh) * Fr * G(wo, wi) / std::abs(4 * cos_theta_o * cos_theta_i);
                DCHECK(!invalid(ret));
                DCHECK(all_positive(ret));
                return ret;
            }

            template<typename T>
            LM_ND_XPU T BRDF(float3 wo, float3 wi, T Fr,
                             float cos_theta_i, float cos_theta_o,
                             TransportMode mode = TransportMode::Radiance) const {
                float3 wh = normalize(wo + wi);
                return BRDF(wo, wh, wi, Fr, cos_theta_i, cos_theta_o, mode);
            }

            /**
             *
             * @tparam T
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            template<typename T>
            LM_ND_XPU T BTDF(float3 wo, float3 wh, float3 wi, T Ft,
                             float cos_theta_i, float cos_theta_o, T eta,
                             TransportMode mode = TransportMode::Radiance) const {
                T numerator = D(wh) * Ft * G(wo, wi) * std::abs(dot(wi, wh) * dot(wo, wh));
                T denom = sqr(dot(wi, wh) * eta + dot(wo, wh)) * abs(cos_theta_i * cos_theta_o);
                T ft = numerator / denom;
                if (mode == TransportMode::Radiance) {
                    ft = ft / sqr(eta);
                }
                DCHECK(!invalid(ft));
                DCHECK(all_positive(ft));
                return ft;
            }

            /**
             *
             * @tparam T
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            template<typename T>
            LM_ND_XPU T BTDF(float3 wo, float3 wi, T Ft,
                             float cos_theta_i, float cos_theta_o, T eta,
                             TransportMode mode = TransportMode::Radiance) const {
                float3 wh = normalize(wo + wi * eta);
                return BTDF(wo, wh, wi, Ft, cos_theta_i, cos_theta_o,eta, mode);
            }
        };
    }
}