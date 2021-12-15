//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        class MicrofacetReflection {
        public:
            MicrofacetReflection() = default;

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, Spectrum R,
                                    TFresnel &fresnel,
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);
                bool reflect = cos_theta_i * cos_theta_o > 0;
                if (!reflect) {
                    return {0.f};
                }
                fresnel.eta = cos_theta_o > 0 ? fresnel.eta : (rcp(fresnel.eta));
                float3 wh = normalize(wo + wi);
                wh = face_forward(wh, make_float3(0, 0, 1));
                auto F = fresnel.eval(dot(wo, wh));
                auto fr = microfacet.BRDF(wo, wh, wi, F, cos_theta_i, cos_theta_o, mode);
                return fr * R;
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

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, Spectrum R,
                                          TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wh = microfacet.sample_wh(wo, u);
                float3 wi = reflect(wo, wh);
                float cos_theta_i = Frame::cos_theta(wi);
                float cos_theta_o = Frame::cos_theta(wo);
                if (!same_hemisphere(wi, wo)) {
                    return {};
                }
                float PDF = microfacet.PDF_wi_reflection(wo, wh);
                Spectrum val = eval(wo, wi, R, fresnel, microfacet);
                return {val, wi, PDF, Reflection, fresnel.eta};
            }
        };

        class MicrofacetTransmission {
        private:
            float3 _t;

        public:
            MicrofacetTransmission() = default;

            explicit MicrofacetTransmission(float3 t) : _t(t) {}

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi,
                                    TFresnel &fresnel,
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                float cos_theta_o = Frame::cos_theta(wo);
                float cos_theta_i = Frame::cos_theta(wi);

            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {

            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {

            }

        };
    }
}