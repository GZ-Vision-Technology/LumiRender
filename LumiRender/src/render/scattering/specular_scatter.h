//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        class SpecularReflection {
        public:
            LM_XPU SpecularReflection() = default;

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TData data,
                                    TFresnel fresnel,
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

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, TData data,
                                          TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {

                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                Spectrum val = fresnel.eval(Frame::cos_theta(wi)) * data.color / Frame::abs_cos_theta(wi);
                float PDF = 1.f;
                return {val, wi, PDF, BxDFFlags::SpecRefl};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecRefl;
            }

            LM_ND_XPU static bool match_flags(BxDFFlags bxdf_flags) {
                return (flags() & bxdf_flags) == flags();
            }
        };

        class SpecularTransmission {
        public:
            LM_XPU SpecularTransmission() = default;

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TData data,
                                    TFresnel fresnel,
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

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, TData data,
                                          TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi{};
                using eta_type = decltype(TFresnel::eta);
                bool valid = refract(wo, make_float3(0, 0, 1), fresnel.eta, &fresnel.eta, &wi);
                if (!valid) {
                    return {};
                }
                auto Fr = fresnel.eval(Frame::cos_theta(wi));
                auto ft = (eta_type(1) - Fr) / Frame::abs_cos_theta(wi);
                eta_type factor = cal_factor(mode, fresnel.eta);
                Spectrum val = ft * data.color * factor;
                return {val, wi, 1, Transmission};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecTrans;
            }

            LM_ND_XPU static bool match_flags(BxDFFlags bxdf_flags) {
                return (flags() & bxdf_flags) == flags();
            }
        };
    }
}