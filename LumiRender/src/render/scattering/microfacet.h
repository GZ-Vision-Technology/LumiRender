//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "flags.h"

namespace luminous {
    inline namespace render {
        enum MicrofacetType : uint8_t {
            None,
            GGX,
            Disney,
            Beckmann,
            NeubeltCloth
        };

        namespace microfacet {
            ND_XPU_INLINE float roughness_to_alpha(float roughness) {
                roughness = std::max(roughness, (float) 1e-3);
                float x = std::log(roughness);
                return 1.62142f +
                       0.819955f * x +
                       0.1734f * Pow<2>(x) +
                       0.0171201f * Pow<3>(x) +
                       0.000640711f * Pow<4>(x);
            }

            /**
             * @brief Cloth Microfacet Distribution from Sony Pictures Imageworks
             * @see http://aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf
             * @param wh half-way vector
             * @param alpha roughness parameter
             * @return Microfacet Distibution PDF
             */
            ND_XPU_INLINE float D_Charlie(const float3 &wh, float alpha) {
                float inv_2alpha = 0.5f / alpha;
                float sin2h = std::max(Frame::cos_theta_2(wh), 0.0078125f);
                return (1.0f + inv_2alpha) * invPi * std::pow(sin2h, inv_2alpha);
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
            ND_XPU_INLINE float D(const float3 &wh, float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                // When theta is close to 90, tan theta is infinity
                float tan_theta_2 = Frame::tan_theta_2(wh);
                if (is_inf(tan_theta_2)) {
                    return 0.f;
                }
                float cos_theta_4 = sqr(Frame::cos_theta_2(wh));
                if (cos_theta_4 < 1e-16f) {
                    return 0.f;
                }
                switch (type) {
                    case Disney:
                    case GGX: {
                        float e = tan_theta_2 * (sqr(Frame::cos_phi(wh) / alpha_x) + sqr(Frame::sin_phi(wh) / alpha_y));
                        float ret = 1.f / (Pi * alpha_x * alpha_y * cos_theta_4 * sqr(1 + e));
                        return ret;
                    }
                    case Beckmann: {
                        return std::exp(-tan_theta_2 * (Frame::cos_phi_2(wh) / sqr(alpha_x) +
                                                        Frame::sin_phi_2(wh) / sqr(alpha_y))) /
                               (Pi * alpha_x * alpha_y * cos_theta_4);
                    }
                    // case NeubeltCloth:
                    //     return D_Charlie(wh, alpha_x);
                    default:
                        break;
                }
                LM_ASSERT(0, "unknown type %d", int(type));
                return 0;
            }

            /**
             * lambda(w) = A-(w) / (A+(w) - A-(w))
             * @param  w [description]
             * @return   [description]
             */
            ND_XPU_INLINE float lambda(const float3 &w, float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                switch (type) {
                    case Disney:
                    case GGX: {
                        float abs_tan_theta = std::abs(Frame::tan_theta(w));
                        if (is_inf(abs_tan_theta)) {
                            return 0.f;
                        }
                        float2 cs_theta_2 = Frame::cos_sin_phi_2(w);
                        float alpha = std::sqrt(cs_theta_2.x * sqr(alpha_x) +
                                                cs_theta_2.y * sqr(alpha_y));
                        float ret = (-1 + std::sqrt(1.f + sqr(alpha * abs_tan_theta))) / 2;
                        return ret;
                    }
                    case Beckmann: {
                        float abs_tan_theta = std::abs(Frame::tan_theta(w));
                        if (is_inf(abs_tan_theta)) {
                            return 0.f;
                        }
                        float2 cs_theta_2 = Frame::cos_sin_phi_2(w);
                        float alpha = std::sqrt(cs_theta_2.x * sqr(alpha_x) +
                                                cs_theta_2.y * sqr(alpha_y));
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
                LM_ASSERT(type != None, "unknown type %d", int(type));
                return 0;
            }

            /**
             * @brief Cloth Shadowing term $ L(x) = a / (1 + b x^c) + dx + e $
             * 
             * @param x cosine of incident(scattering) vector
             * @return L values for roughness parameter in point 0.0 and 1.0
             */
            ND_XPU_INLINE float2 L_Neubelt(float x) {
                constexpr float2 a = float2(25.3245f, 21.5473f);
                constexpr float2 b = float2(3.32435f, 3.82987f);
                constexpr float2 c = float2(0.16801f, 0.19823f);
                constexpr float2 d = float2(-1.27393f, -1.97760f);
                constexpr float2 e = float2(-4.85967f, -4.32054f);

                float2 x_pc = float2(std::pow(x, c.x), std::pow(x, c.y));
                return a / (float2(1.0f, 1.0f) + b * x_pc) + d * float2(x, x) + e;
            }

            ND_XPU_INLINE float2 Lambda_border_Neubelt(float x) {
                constexpr float2 _2t_Lhalf = float2(1.800426f, -0.6849023f);
                if(x <= 0.5f) {
                    float2 L = L_Neubelt(x);
                    return float2(std::exp(L.x), std::exp(L.y));
                } else {
                    float2 L = _2t_Lhalf - L_Neubelt(1.0f - x);
                    return float2(std::exp(L.x), std::exp(L.y));
                }
            }

            ND_XPU_INLINE float G_Neubelt(const float3 &wo, const float3 &wi, float alpha) {

                float cos_theta_o = Frame::cos_theta(wo), cos_theta_i = Frame::cos_theta(wi);
                float2 lambda_border_o = Lambda_border_Neubelt(cos_theta_o);
                float2 lambda_border_i = Lambda_border_Neubelt(cos_theta_i);

                float t = sqr(1.0f - alpha);
                float lambda_o = lerp(t, lambda_border_o.y, lambda_border_o.x);
                float lambda_i = lerp(t, lambda_border_i.y, lambda_border_i.x);
                return 1.0f / (1.0f + lambda_o + lambda_i);
            }

            ND_XPU_INLINE float G_Neubelt_soften(const float3 &wo, const float3 &wi, float alpha) {

                float cos_theta_o = Frame::cos_theta(wo), cos_theta_i = Frame::cos_theta(wi);
                float2 lambda_border_o = Lambda_border_Neubelt(cos_theta_o);
                float2 lambda_border_i = Lambda_border_Neubelt(cos_theta_i);

                float t = sqr(1.0f - alpha);
                float lambda_o = lerp(t, lambda_border_o.y, lambda_border_o.x);
                float lambda_i = std::pow(lerp(t, lambda_border_i.y, lambda_border_i.x),
                        1.0f + Pow<8>(1 - cos_theta_i));
                return 1.0f / (1.0f + lambda_o + lambda_i);
            }

            /**
             * smith occlusion function
             * G1(w) = 1 / (lambda(w) + 1)
             * @param  w [description]
             * @return   [description]
             */
            ND_XPU_INLINE float G1(const float3 &w, float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                float ret = 1 / (1 + lambda(w, alpha_x, alpha_y, type));
                return ret;
            }

            /**
             * G(wo, wi) = 1 / (lambda(wo) + lambda(wi) + 1)
             * @return   [description]
             */
            ND_XPU_INLINE float
            G(const float3 &wo, const float3 &wi, float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                float ret = 0.f;
                switch (type) {
                    case Disney: {
                        ret = G1(wi, alpha_x, alpha_y, type) * G1(wo, alpha_x, alpha_y, type);
                        return ret;
                    }
                    case GGX:
                    case Beckmann: {
                        ret = 1 / (1 + lambda(wo, alpha_x, alpha_y, type) + lambda(wi, alpha_x, alpha_y, type));
                        return ret;
                    }
                    // case NeubeltCloth:
                        // return G_Neubelt_soften(wo, wi, alpha_x);
                    default:
                        break;
                }
                LM_ASSERT(type != None, "unknown type %d", int(type));
                return ret;
            }

            ND_XPU_INLINE float3 sample_wh(const float3 &wo, const float2 &u,
                                           float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                switch (type) {
                    case Disney:
                    case GGX: {
                        float cos_theta = 0, phi = _2Pi * u[1];
                        if (alpha_x == alpha_y) {
                            float tan_theta_2 = alpha_x * alpha_x * u[0] / (1.0f - u[0]);
                            cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        } else {
                            phi = std::atan(alpha_y / alpha_x * std::tan(_2Pi * u[1] + PiOver2));
                            if (u[1] > .5f) {
                                phi += Pi;
                            }
                            float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                            const float alpha2 = 1.f / (sqr(cos_phi / alpha_x) + sqr(sin_phi / alpha_y));
                            float tan_theta_2 = alpha2 * u[0] / (1 - u[0]);
                            cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        }
                        float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                        float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                        if (!same_hemisphere(wo, wh)) {
                            wh = -wh;
                        }CHECK_UNIT_VEC(wh);
                        return wh;
                    }
                    case Beckmann: {
                        float tan_theta_2, phi;
                        if (alpha_x == alpha_y) {
                            float log_sample = std::log(1 - u[0]);
                            DCHECK(!is_inf(log_sample));
                            tan_theta_2 = -alpha_x * alpha_x * log_sample;
                            phi = u[1] * _2Pi;
                        } else {
                            float log_sample = std::log(1 - u[0]);
                            DCHECK(!is_inf(log_sample));
                            phi = std::atan(alpha_y / alpha_x *
                                            std::tan(_2Pi * u[1] + PiOver2));
                            if (u[1] > 0.5f) {
                                phi += Pi;
                            }
                            float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                            tan_theta_2 = -log_sample / (sqr(cos_phi / alpha_x) + sqr(sin_phi / alpha_y));
                        }

                        float cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                        float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                        float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                        if (!same_hemisphere(wo, wh)) {
                            wh = -wh;
                        }CHECK_UNIT_VEC(wh);
                        return wh;
                    }
                    // case NeubeltCloth:
                    //     {
                    //         // Uniformly sample wh on hemisphere
                    //         float3 wh = square_to_hemisphere(u);
                    //         if(!same_hemisphere(wo, wh))
                    //             wh = -wh;
                    //         return wh;
                    //     }
                    default:
                        break;
                }
                LM_ASSERT(type != None, "unknown type %d", int(type));
                return {};
            }

            /**
            * @param  wo
            * @param  wh :normal of microfacet
            * @return
            */
            ND_XPU_INLINE float PDF_wh(const float3 &wo, const float3 &wh,
                                       float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                // if(type == NeubeltCloth)
                //     return uniform_hemisphere_PDF();
                // else 
                    return D(wh, alpha_x, alpha_y, type) * Frame::abs_cos_theta(wh);
            }

            /**
             * pwi(wi) = dwh / dwi * pwh(wh) = pwh(wh) / 4cos_theta_h
             * @param PDF_wh
             * @param wo
             * @param wh
             * @return
             */
            ND_XPU_INLINE float PDF_wi_reflection(float PDF_wh, float3 wo, float3 wh) {
                float ret = PDF_wh / (4 * abs_dot(wo, wh));
                DCHECK(!is_invalid(ret));
                return ret;
            }

            ND_XPU_INLINE float PDF_wi_reflection(float3 wo, float3 wh,
                                                  float alpha_x, float alpha_y, MicrofacetType type = GGX) {
                return PDF_wi_reflection(PDF_wh(wo, wh, alpha_x, alpha_y, type), wo, wh);
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
            ND_XPU_INLINE float PDF_wi_transmission(float PDF_wh, float3 wo, float3 wh, float3 wi, float eta) {
                float denom = sqr(dot(wi, wh) * eta + dot(wo, wh));
                float dwh_dwi = abs_dot(wi, wh) / denom;
                float ret = PDF_wh * dwh_dwi;
                DCHECK(!is_invalid(ret));
                return ret;
            }

            ND_XPU_INLINE Spectrum BRDF(float3 wo, float3 wh, float3 wi, Spectrum Fr,
                                        float cos_theta_i, float cos_theta_o,
                                        float alpha_x, float alpha_y, MicrofacetType type = GGX,
                                        TransportMode mode = TransportMode::Radiance) {
                auto ret = D(wh, alpha_x, alpha_y, type) * Fr * G(wo, wi, alpha_x, alpha_y, type)
                           / std::abs(4 * cos_theta_o * cos_theta_i);
                DCHECK(!invalid(ret));
                DCHECK(all_positive(ret));
                return ret;
            }

            ND_XPU_INLINE Spectrum BRDF(float3 wo, float3 wi, Spectrum Fr,
                                        float cos_theta_i, float cos_theta_o,
                                        float alpha_x, float alpha_y,MicrofacetType type = GGX,
                                        TransportMode mode = TransportMode::Radiance) {
                float3 wh = normalize(wo + wi);
                return microfacet::BRDF(wo, wh, wi, Fr, cos_theta_i, cos_theta_o, alpha_x, alpha_y, type, mode);
            }

            /**
             *
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            ND_XPU_INLINE Spectrum BTDF(float3 wo, float3 wh, float3 wi, Spectrum Ft,
                                     float cos_theta_i, float cos_theta_o, float eta,
                                     float alpha_x, float alpha_y, MicrofacetType type = GGX,
                                     TransportMode mode = TransportMode::Radiance) {
                Spectrum numerator = D(wh, alpha_x, alpha_y, type) * Ft * G(wo, wi, alpha_x, alpha_y, type) *
                                  std::abs(dot(wi, wh) * dot(wo, wh));
                float denom = sqr(dot(wi, wh) * eta + dot(wo, wh)) * abs(cos_theta_i * cos_theta_o);
                Spectrum ft = numerator / denom;
                float factor = cal_factor(mode, eta);
                DCHECK(!invalid(ft));
                DCHECK(all_positive(ft));
                return ft * factor;
            }

            /**
             *
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            ND_XPU_INLINE Spectrum BTDF(float3 wo, float3 wi, Spectrum Ft,
                                     float cos_theta_i, float cos_theta_o, float eta,
                                     float alpha_x, float alpha_y, MicrofacetType type = GGX,
                                     TransportMode mode = TransportMode::Radiance) {
                float3 wh = normalize(wo + wi * eta);
                return BTDF(wo, wh, wi, Ft, cos_theta_i, cos_theta_o, eta, alpha_x, alpha_y, type, mode);
            }
        }

        class Microfacet {
        public:
            float _alpha_x{};
            float _alpha_y{};
            MicrofacetType _type{None};
        public:
            LM_XPU Microfacet() = default;

            LM_XPU explicit Microfacet(float alpha, MicrofacetType type = GGX)
                    : _alpha_x(alpha),
                      _alpha_y(alpha),
                      _type(type) {
                LM_ASSERT(_type != None, "unknown type %d", int(_type));
            }

            LM_XPU Microfacet(float alpha_x, float alpha_y, MicrofacetType type = GGX)
                    : _alpha_x(alpha_x),
                      _alpha_y(alpha_y),
                      _type(type) {
                LM_ASSERT(_type != None, "unknown type %d", int(_type));
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

            LM_ND_XPU float max_alpha() const {
                return std::max(_alpha_x, _alpha_y);
            }

            LM_ND_XPU bool effectively_smooth() const {
                return std::max(_alpha_x, _alpha_y) < 1e-3f;
            }

            LM_ND_XPU float D(const float3 &wh) const {
                return microfacet::D(wh, _alpha_x, _alpha_y, _type);
            }

            // LM_ND_XPU float lambda(const float3 &w) const {
            //     return microfacet::lambda(w, _alpha_x, _alpha_y, _type);
            // }

            // LM_ND_XPU float G1(const float3 &w) const {
            //     return microfacet::G1(w, _alpha_x, _alpha_y, _type);
            // }

            // LM_ND_XPU float G(const float3 &wo, const float3 &wi) const {
            //     return microfacet::G(wo, wi, _alpha_x, _alpha_y, _type);
            // }

            LM_ND_XPU float3 sample_wh(const float3 &wo, const float2 &u) const {
                return microfacet::sample_wh(wo, u, _alpha_x, _alpha_y, _type);
            }

            LM_ND_XPU float PDF_wh(const float3 &wo, const float3 &wh) const {
                return microfacet::PDF_wh(wo, wh, _alpha_x, _alpha_y, _type);
            }

            /**
             * pwi(wi) = dwh / dwi * pwh(wh) = pwh(wh) / 4cos_theta_h
             * @param PDF_wh
             * @param wo
             * @param wh
             * @return
             */
            LM_ND_XPU float PDF_wi_reflection(float PDF_wh, float3 wo, float3 wh) const {
                return microfacet::PDF_wi_reflection(PDF_wh, wo, wh);
            }

            LM_ND_XPU float PDF_wi_reflection(float3 wo, float3 wh) const {
                return microfacet::PDF_wi_reflection(PDF_wh(wo, wh), wo, wh);
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
                return microfacet::PDF_wi_transmission(PDF_wh, wo, wh, wi, eta);
            }

            LM_ND_XPU float PDF_wi_transmission(float3 wo, float3 wh, float3 wi, float eta) const {
                return PDF_wi_transmission(PDF_wh(wo, wh), wo, wh, wi, eta);
            }

            LM_ND_XPU Spectrum BRDF(float3 wo, float3 wh, float3 wi, Spectrum Fr,
                                    float cos_theta_i, float cos_theta_o,
                                    TransportMode mode = TransportMode::Radiance) const {
                return microfacet::BRDF(wo, wh, wi, Fr, cos_theta_i, cos_theta_o, _alpha_x, _alpha_y, _type, mode);
            }

            LM_ND_XPU Spectrum BRDF(float3 wo, float3 wi, Spectrum Fr,
                                    float cos_theta_i, float cos_theta_o,
                                    TransportMode mode = TransportMode::Radiance) const {
                float3 wh = normalize(wo + wi);
                return BRDF(wo, wh, wi, Fr, cos_theta_i, cos_theta_o, mode);
            }

            /**
             *
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            LM_ND_XPU Spectrum BTDF(float3 wo, float3 wh, float3 wi, Spectrum Ft,
                                 float cos_theta_i, float cos_theta_o, float eta,
                                 TransportMode mode = TransportMode::Radiance) const {
                return microfacet::BTDF(wo, wh, wi, Ft, cos_theta_i, cos_theta_o, eta, _alpha_x, _alpha_y, _type, mode);
            }

            /**
             *
             * @param eta : eta_i / eta_o
             * @param mode
             * @return
             */
            LM_ND_XPU Spectrum BTDF(float3 wo, float3 wi, Spectrum Ft,
                                 float cos_theta_i, float cos_theta_o, float eta,
                                 TransportMode mode = TransportMode::Radiance) const {
                float3 wh = normalize(wo + wi * eta);
                return BTDF(wo, wh, wi, Ft, cos_theta_i, cos_theta_o, eta, mode);
            }
        };
    }
}