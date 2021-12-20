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

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, TData data,
                                     TFresnel fresnel,
                                     TMicrofacet microfacet = {},
                                     TransportMode mode = TransportMode::Radiance) const {
                // must be reflection and eta must be correct
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                float3 wh = normalize(wo + wi);
                wh = face_forward(wh, make_float3(0, 0, 1));
                auto F = fresnel.eval(dot(wo, wh));
                auto fr = microfacet.BRDF(wo, wh, wi, F, cos_theta_i, cos_theta_o, mode);
                return fr * data.color;
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TData data,
                                    TFresnel fresnel,
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                float cos_theta_o = Frame::cos_theta(wo);
                if (!same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                fresnel.correct_eta(cos_theta_o);
                return _eval(wo, wi, data, fresnel, microfacet, mode);
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                if (!same_hemisphere(wo, wi)) {
                    return 0.f;
                }
                float3 wh = normalize(wo + wi);
                return microfacet.PDF_wi_reflection(wo, wh);
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, TData data,
                                          TFresnel fresnel, TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wh = microfacet.sample_wh(wo, u);
                float3 wi = reflect(wo, wh);
                float cos_theta_o = Frame::cos_theta(wo);
                if (!same_hemisphere(wi, wo)) {
                    return {};
                }
                fresnel.correct_eta(cos_theta_o);
                float PDF = microfacet.PDF_wi_reflection(wo, wh);
                Spectrum val = _eval(wo, wi, data, fresnel, microfacet);
                return {val, wi, PDF, Reflection, fresnel.eta};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyReflection;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class MicrofacetTransmission : public BxDF {
        public:
            using BxDF::BxDF;

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, TData data,
                                     TFresnel fresnel,
                                     TMicrofacet microfacet = {},
                                     TransportMode mode = TransportMode::Radiance) const {
                // must be transmission and eta must be correct
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                using eta_type = decltype(fresnel.eta);
                float3 wh = normalize(wo + wi * fresnel.eta);
                wh = face_forward(wh, make_float3(0, 0, 1));
                eta_type F = fresnel.eval(dot(wo, wh));
                eta_type tr = microfacet.BTDF(wo, wh, wi, eta_type(1.f) - F, cos_theta_i, cos_theta_o, fresnel.eta, mode);
                return tr * data.color;
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TData data,
                                    TFresnel fresnel,
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                float cos_theta_o = Frame::cos_theta(wo);
                if (same_hemisphere(wi, wo)) {
                    return {0.f};
                }
                fresnel.correct_eta(cos_theta_o);
                return _eval(wo, wi, data, fresnel, microfacet, mode);
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                if (same_hemisphere(wo, wi)) {
                    return 0.f;
                }
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                fresnel.correct_eta(cos_theta_o);
                float3 wh = normalize(wo + wi * fresnel.eta);
                return microfacet.PDF_wi_transmission(wo, wh, wi, fresnel.eta);
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, TData data,
                                          TFresnel fresnel, TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wh = microfacet.sample_wh(wo, u);
                float3 wi{};
                bool valid = refract(wo, wh, fresnel.eta, &fresnel.eta, &wi);
                if (!valid || same_hemisphere(wo, wi)) {
                    return {};
                }
                float cos_theta_o = Frame::cos_theta(wo);
                fresnel.correct_eta(cos_theta_o);
                float PDF = microfacet.PDF_wi_transmission(wo, wh, wi, fresnel.eta);
                Spectrum val = _eval(wo, wi, data, fresnel, microfacet);
                return {val, wi, PDF, Transmission, fresnel.eta};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyTransmission;
            }

            GEN_MATCH_FLAGS_FUNC
        };
    }
}