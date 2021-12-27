//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"

namespace luminous {
    inline namespace render {

        class MicrofacetReflection : public BxDF {
        public:
            using BxDF::BxDF;

            /**
             * must be reflection and eta must be corrected
             */
            LM_ND_XPU static Spectrum _eval(float3 wo, float3 wi, BSDFData data,
                                            Fresnel fresnel,
                                            Microfacet microfacet = {},
                                            TransportMode mode = TransportMode::Radiance) {

                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                float3 wh = normalize(wo + wi);
                wh = face_forward(wh, make_float3(0, 0, 1));
                auto F = fresnel.eval(abs_dot(wo, wh), data);
                auto fr = microfacet.BRDF(wo, wh, wi, F, cos_theta_i, cos_theta_o, mode);
                return fr * Spectrum(data.color());
            }

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                if (!same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                data.correct_eta(cos_theta_o, fresnel.type());
                return _eval(wo, wi, data, fresnel, microfacet, mode);
            }

            /**
            * must be reflection
            */
            LM_ND_XPU static float _PDF(float3 wo, float3 wi,
                                        BSDFData data,
                                        Fresnel fresnel,
                                        Microfacet microfacet = {},
                                        TransportMode mode = TransportMode::Radiance) {
                float3 wh = normalize(wo + wi);
                return microfacet.PDF_wi_reflection(wo, wh);
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                if (!same_hemisphere(wo, wi)) {
                    return 0.f;
                }
                return _PDF(wo, wi, data, fresnel, microfacet, mode);
            }

            /**
             * eta must be corrected
             */
            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  Spectrum Fr, BSDFData data,
                                                  Fresnel fresnel,
                                                  Microfacet microfacet = {},
                                                  TransportMode mode = TransportMode::Radiance) {
                float3 wh = microfacet.sample_wh(wo, u);
                if (dot(wh, wo) < 0) {
                    return {};
                }
                float3 wi = reflect(wo, wh);
                if (!same_hemisphere(wi, wo)) {
                    return {};
                }
                float PDF = microfacet.PDF_wi_reflection(wo, wh);
                Spectrum val = _eval(wo, wi, data, fresnel, microfacet);
                return {val, wi, PDF, flags(), data.eta()};
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel, Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                return _sample_f(wo, uc, u, 0.f, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class MicrofacetTransmission : public BxDF {
        public:
            using BxDF::BxDF;

            /**
             * must be transmission and eta must be corrected
             */
            LM_ND_XPU static Spectrum _eval(float3 wo, float3 wi, BSDFData data,
                                            Fresnel fresnel,
                                            Microfacet microfacet = {},
                                            TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                using eta_type = decltype(data.eta());
                float3 wh = normalize(wo + wi * data.eta());
                if (dot(wo, wh) * dot(wi, wh) > 0) {
                    return {0.f};
                }
                wh = face_forward(wh, make_float3(0, 0, 1));
                float F = fresnel.eval(abs_dot(wo, wh), data)[0];
                float tr = microfacet.BTDF(wo, wh, wi, eta_type(1.f) - F, cos_theta_i, cos_theta_o, data.eta(),
                                           mode);
                return tr * data.color();
            }

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                if (same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                data.correct_eta(cos_theta_o, fresnel.type());
                return _eval(wo, wi, data, fresnel, microfacet, mode);
            }

            /**
             * wo and wi must be not same hemisphere
             * and eta must be corrected
             */
            LM_ND_XPU static float _PDF(float3 wo, float3 wi,
                                        BSDFData data,
                                        Fresnel fresnel,
                                        Microfacet microfacet,
                                        TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                float3 wh = normalize(wo + wi * data.eta());
                if (dot(wo, wh) * dot(wi, wh) > 0) {
                    return 0.f;
                }
                wh = face_forward(wh, make_float3(0, 0, 1));
                return microfacet.PDF_wi_transmission(wo, wh, wi, data.eta());
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet,
                                       TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                if (same_hemisphere(wo, wi)) {
                    return 0.f;
                }
                data.correct_eta(cos_theta_o, fresnel.type());
                return _PDF(wo, wi, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  Spectrum Fr, BSDFData data,
                                                  Fresnel fresnel,
                                                  Microfacet microfacet = {},
                                                  TransportMode mode = TransportMode::Radiance) {
                float3 wh = microfacet.sample_wh(wo, u);
                if (dot(wh, wo) < 0) {
                    return {};
                }
                float3 wi{};
                bool valid = refract(wo, wh, data.eta(), &wi);
                if (!valid || same_hemisphere(wo, wi)) {
                    return {};
                }
                float PDF = microfacet.PDF_wi_transmission(wo, wh, wi, data.eta());
                Spectrum val = _eval(wo, wi, data, fresnel, microfacet);
                return {val, wi, PDF, flags(), data.eta()};
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel, Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                return _sample_f(wo, uc, u, 0.f, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyTrans;
            }

            GEN_MATCH_FLAGS_FUNC
        };


        class MicrofacetFresnel : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                if (same_hemisphere(wi, wo)) {
                    return MicrofacetReflection::_eval(wo, wi, data, fresnel, microfacet, mode);
                }
                return MicrofacetTransmission::_eval(wo, wi, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                if (same_hemisphere(wi, wo)) {
                    return MicrofacetReflection::_PDF(wo, wi, data, fresnel, microfacet, mode);
                }
                return MicrofacetTransmission::_PDF(wo, wi, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel,
                                                 Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                float Fr = fresnel.eval(Frame::abs_cos_theta(wo), data)[0];
                BSDFSample ret;
                if (uc < Fr) {
                    // sample reflection
                    ret = MicrofacetReflection::_sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
                    ret.PDF *= Fr;
                } else {
                    // sample transmission
                    ret = MicrofacetTransmission::_sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
                    ret.PDF *= 1 - Fr;
                }
                return ret;
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyTrans | BxDFFlags::GlossyRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };
    }
}