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

        template<MicrofacetType microfacet_type = GGX>
        class Microfacet {
        private:
            float _alpha_x{};
            float _alpha_y{};
            constexpr static MicrofacetType _type{microfacet_type};
        public:
            LM_XPU Microfacet() = default;

            LM_XPU Microfacet(float alpha_x, float alpha_y)
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
            LM_ND_XPU float D(const float3 &wh) const {
                // When theta is close to 90, tan theta is infinity
                float tan_theta_2 = Frame::tan_theta_2(wh);
                if (is_inf(tan_theta_2)) {
                    return 0.f;
                }
                float cos_theta_4 = sqr(Frame::cos_theta_2(wh));
                if (cos_theta_4 < 1e-16f) {
                    return 0.f;
                }
                switch (_type) {
                    case GGX: {
                        float e =
                                tan_theta_2 * (sqr(Frame::cos_phi(wh) / _alpha_x) + sqr(Frame::sin_phi(wh) / _alpha_y));
                        float ret = 1.f / (Pi * _alpha_x * _alpha_y * cos_theta_4 * sqr(1 + e));
                        return ret;
                    }
                    case Beckmann: {
                        return std::exp(-tan_theta_2 * (Frame::cos_phi_2(wh) / sqr(_alpha_x) +
                                                        Frame::sin_phi_2(wh) / sqr(_alpha_y))) /
                               (Pi * _alpha_x * _alpha_y * cos_theta_4);
                    }
                    default:
                        break;
                }
                LM_ASSERT(0, "unknown type %d", int(_type));
                return 0;
            }

            /**
             * lambda(w) = A-(w) / (A+(w) - A-(w))
             * @param  w [description]
             * @return   [description]
             */
            LM_ND_XPU float lambda(const float3 &w) const {
                switch (_type) {
                    case GGX: {
                        float abs_tan_theta = std::abs(Frame::tan_theta(w));
                        if (is_inf(abs_tan_theta)) {
                            return 0.f;
                        }
                        float alpha = std::sqrt(Frame::cos_phi_2(w) * sqr(_alpha_x) +
                                                Frame::sin_phi_2(w) * sqr(_alpha_y));
                        float ret = (-1 + std::sqrt(1.f + sqr(alpha * abs_tan_theta))) / 2;
                        return ret;
                    }
                    case Beckmann: {
                        float abs_tan_theta = std::abs(Frame::tan_theta(w));
                        if (is_inf(abs_tan_theta)) {
                            return 0.f;
                        }
                        float alpha = std::sqrt(Frame::cos_phi_2(w) * sqr(_alpha_x) +
                                                Frame::sin_phi_2(w) * sqr(_alpha_y));
                        float a = 1.f / (alpha * abs_tan_theta);
                        if (a >= 1.6f) {
                            return 0.f;
                        }
                        float ret = (1 - 1.259f * a + 0.396f * sqr(a)) / (3.535f * a + 2.181f * sqr(a));
                        return ret;
                    }
                    default:
                        break;
                }
                LM_ASSERT(0, "unknown type %d", int(_type));
                return 0;
            }

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

            LM_ND_XPU float3 sample_wh(const float3 &wo, const float2 &u) const {
                switch (_type) {
                    case GGX: {
                        float cos_theta = 0, phi = _2Pi * u[1];
                        if (_alpha_x == _alpha_y) {
                            float tan_theta_2 = _alpha_x * _alpha_x * u[0] / (1.0f - u[0]);
                            cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        } else {
                            phi = std::atan(_alpha_y / _alpha_x * std::tan(_2Pi * u[1] + PiOver2));
                            if (u[1] > .5f) {
                                phi += Pi;
                            }
                            float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                            const float alpha2 = 1.f / (sqr(cos_phi / _alpha_x) + sqr(sin_phi / _alpha_y));
                            float tan_theta_2 = alpha2 * u[0] / (1 - u[0]);
                            cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        }
                        float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                        float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                        if (!same_hemisphere(wo, wh)) {
                            wh = -wh;
                        }
                        return wh;
                    }
                    case Beckmann: {
                        float tan_theta_2, phi;
                        if (_alpha_x == _alpha_y) {
                            float log_sample = std::log(1 - u[0]);
                            DCHECK(!is_inf(log_sample));
                            tan_theta_2 = -_alpha_x * _alpha_x * log_sample;
                            phi = u[1] * _2Pi;
                        } else {
                            float log_sample = std::log(1 - u[0]);
                            DCHECK(!is_inf(log_sample));
                            phi = std::atan(_alpha_y / _alpha_x *
                                            std::tan(_2Pi * u[1] + PiOver2));
                            if (u[1] > 0.5f) {
                                phi += Pi;
                            }
                            float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                            tan_theta_2 = -log_sample / (sqr(cos_phi / _alpha_x) + sqr(sin_phi / _alpha_y));
                        }

                        float cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                        float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                        if (!same_hemisphere(wo, wh)) {
                            wh = -wh;
                        }
                        return wh;
                    }
                    default:
                        break;
                }
                LM_ASSERT(0, "unknown type %d", int(_type));
                return {};
            }

            /**
             * @param  wo
             * @param  wh :normal of microfacet
             * @return
             */
            LM_ND_XPU float PDF_wh(const float3 &wo, const float3 &wh) const {
                return D(wh) * Frame::abs_cos_theta(wh);
            }

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
                return BTDF(wo, wh, wi, Ft, cos_theta_i, cos_theta_o, eta, mode);
            }
        };
    }
}