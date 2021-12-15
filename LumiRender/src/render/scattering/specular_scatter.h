//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        class SpecularReflection {
        private:
            float3 _r;
        public:
            LM_XPU explicit SpecularReflection(float3 r)
                    : _r(r) {}

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi,
                                    TFresnel fresnel = {},
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                Spectrum val = fresnel.eval(Frame::cos_theta(wi)) * _r / Frame::abs_cos_theta(wi);
                float PDF = 1.f;
                return {val, wi, PDF, BxDFFlags::SpecRefl};
            }
        };

        class SpecularTransmission {
        private:
            float3 _t;
        public:
            LM_XPU explicit SpecularTransmission(float3 t)
                    : _t(t) {}

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi,
                                    TFresnel fresnel = {},
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi{};
                decltype(TFresnel::eta) eta_p{0.f};
                bool valid = refract(wo, make_float3(0, 0, 1), fresnel.eta, &eta_p, &wi);
                if (!valid) {
                    return {};
                }
                auto R = fresnel.eval(Frame::cos_theta(wi));
                auto T = 1 - R;
                auto ft = T / Frame::abs_cos_theta(wi);
                if (mode == TransportMode::Radiance) {
                    ft = ft / sqr(eta_p);
                }
                Spectrum val = ft * _t;
                return {val, wi, 1, Transmission};
            }
        };
    }
}