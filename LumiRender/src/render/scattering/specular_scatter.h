//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"

namespace luminous {
    inline namespace render {

        class SpecularReflection : public ColoredBxDF<SpecularReflection> {
        public:
            using ColoredBxDF::ColoredBxDF;

            LM_XPU explicit SpecularReflection(Spectrum color) : ColoredBxDF(color, SpecRefl) {}

            ND_XPU_INLINE float weight(BSDFHelper helper, float Fr) const {
                return ColoredBxDF::weight(helper, Fr) * Fr;
            }

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                     BSDFHelper helper,
                                     TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                           Spectrum Fr, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;
        };

        class SpecularTransmission : public ColoredBxDF<SpecularTransmission> {
        public:
            using ColoredBxDF::ColoredBxDF;

            LM_XPU explicit SpecularTransmission(Spectrum color) : ColoredBxDF(color, SpecTrans) {}

            ND_XPU_INLINE float weight(BSDFHelper helper, float Fr) const {
                return ColoredBxDF::weight(helper, Fr) * (1 - Fr);
            }

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                     BSDFHelper helper,
                                     TransportMode mode = TransportMode::Radiance) const {
                return 0.f;
            }

            LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                           Spectrum Fr, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;
        };
    }
}