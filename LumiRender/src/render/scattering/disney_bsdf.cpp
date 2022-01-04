//
// Created by Zero on 27/12/2021.
//

#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {
        namespace disney {

            ND_XPU_INLINE float GTR1(float cos_theta, float alpha) {
                float alpha2 = sqr(alpha);
                return (alpha2 - 1) /
                       (Pi * std::log(alpha2) * (1 + (alpha2 - 1) * sqr(cos_theta)));
            }

            ND_XPU_INLINE float smithG_GGX(float cos_theta, float alpha) {
                float alpha2 = sqr(alpha);
                float cos_theta_2 = sqr(cos_theta);
                return 1 / (cos_theta + sqrt(alpha2 + cos_theta_2 - alpha2 * cos_theta_2));
            }

            Spectrum Diffuse::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                return _factor * helper.color() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
            }

            Spectrum FakeSS::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fss90 = sqr(cos_theta_d) * helper.roughness();
                float Fo = schlick_weight(Frame::abs_cos_theta(wo)),
                        Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Fss = lerp(Fo, 1.f, Fss90) * lerp(Fi, 1.f, Fss90);
                float ss = 1.25f * (Fss * (1 / (Frame::abs_cos_theta(wo) + Frame::abs_cos_theta(wi)) - .5f) + .5f);

                return helper.color() * invPi * ss * _factor;
            }

            Spectrum Retro::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Rr = 2 * helper.roughness() * sqr(cos_theta_d);

                return helper.color() * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1)) * _factor;
            }

            Spectrum Sheen::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                return helper.color_sheen_tint() * _factor * schlick_weight(cos_theta_d);
            }

            // Clearcoat
            Spectrum Clearcoat::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.gloss());
                float Fr = fresnel_schlick(0.04f, dot(wo, wh));
                float Gr = smithG_GGX(Frame::abs_cos_theta(wo), 0.25f)
                           * smithG_GGX(Frame::abs_cos_theta(wi), 0.25f);
                return _weight * Gr * Fr * Dr / 4;
            }

            float Clearcoat::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return 0.f;
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.gloss());
                return Dr * Frame::abs_cos_theta(wh) / (4 * dot(wo, wh));
            }

            float Clearcoat::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                if (!same_hemisphere(wo, wi)) {
                    return 0;
                }
                return PDF(wo, wi, helper, mode);
            }

            BSDFSample Clearcoat::sample_f(float3 wo, float uc, float2 u,
                                           BSDFHelper helper, TransportMode mode) const {
                if (wo.z == 0) {
                    return {};
                }
                float gloss = helper.gloss();
                float alpha2 = sqr(gloss);

                float cos_theta = safe_sqrt((1 - std::pow(alpha2, 1 - u[0])) / (1 - alpha2));
                float sin_theta = safe_sqrt(1 - sqr(cos_theta));
                float phi = 2 * Pi * u[1];
                float3 wh = spherical_direction(sin_theta, cos_theta, phi);

                wh = same_hemisphere(wo, wh) ? wh : -wh;
                float3 wi = reflect(wo, wh);
                if (!same_hemisphere(wo, wi)) {
                    return {};
                }
                float pdf = PDF(wo, wi, helper, mode);
                Spectrum f_val = eval(wo, wi, helper, mode);
                return {f_val, wi, pdf, BxDFFlags::GlossyRefl};
            }

            // SpecularTransmission
            BSDFSample SpecularTransmission::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr, BSDFHelper helper,
                                                       TransportMode mode) const {
                float3 wi{};
                float3 n = make_float3(0, 0, 1);
                bool valid = refract(wo, face_forward(n, wo), helper.eta(), &wi);
                if (!valid) {
                    return {};
                }
                Spectrum ft = (Spectrum(1.f) - Fr) / Frame::abs_cos_theta(wi);
                float factor = cal_factor(mode, helper.eta());
                Spectrum val = ft * factor; // * Spectrum(1.f)
                return {val, wi, 1, SpecTrans, helper.eta()};
            }

            BSDFSample SpecularTransmission::sample_f(float3 wo, float uc, float2 u,
                                                      BSDFHelper helper, TransportMode mode) const {
                float3 wi{};
                helper.correct_eta(Frame::cos_theta(wo));
                float3 n = make_float3(0, 0, 1);
                bool valid = refract(wo, face_forward(n, wo), helper.eta(), &wi);
                if (!valid) {
                    return {};
                }
                auto Fr = helper.eval_fresnel(Frame::abs_cos_theta(wo))[0];
                return _sample_f(wo, uc, u, Fr, helper, mode);
            }

            // MicrofacetReflection
            Spectrum MicrofacetReflection::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return _f(wo, wi, helper, helper.color(), mode);
            }

            Spectrum MicrofacetReflection::safe_eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float cos_theta_o = Frame::cos_theta(wo);
                if (!same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                helper.correct_eta(cos_theta_o);
                return eval(wo, wi, helper, mode);
            }

            BSDFSample MicrofacetReflection::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr,
                                                       BSDFHelper helper, TransportMode mode) const {
                return _sample_f_color(wo, uc, u, Fr, helper, make_float4(1.f), mode);
            }

            BSDFSample MicrofacetReflection::sample_f(float3 wo, float uc, float2 u,
                                                      BSDFHelper helper, TransportMode mode) const {
                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                return _sample_f(wo, uc, u, 0.f, helper, mode);
            }

            // MicrofacetTransmission
            Spectrum MicrofacetTransmission::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return _f(wo, wi, helper, color(helper), mode);
            }

            Spectrum MicrofacetTransmission::safe_eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float cos_theta_o = Frame::cos_theta(wo);
                if (same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                helper.correct_eta(cos_theta_o);
                return eval(wo, wi, helper, mode);
            }

            BSDFSample MicrofacetTransmission::_sample_f(float3 wo, float uc, float2 u,
                                                         Spectrum Fr, BSDFHelper helper,
                                                         TransportMode mode) const {
                return _sample_f_color(wo, uc, u, Fr, helper, helper.spec_trans() * sqrt(helper.color()), mode);
            }

            BSDFSample MicrofacetTransmission::sample_f(float3 wo, float uc, float2 u,
                                                        BSDFHelper helper, TransportMode mode) const {
                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                return _sample_f(wo, uc, u, 0.f, helper, mode);
            }
        }
    }
}