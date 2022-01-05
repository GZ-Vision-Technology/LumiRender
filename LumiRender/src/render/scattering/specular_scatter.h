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