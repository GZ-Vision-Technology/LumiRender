//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"

namespace luminous {
    inline namespace render {

        class MicrofacetReflection : public BxDF<MicrofacetReflection> {
        public:
            using BxDF::BxDF;

            /**
             * must be reflection and eta must be corrected
             */
            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                                TransportMode mode = TransportMode::Radiance);

            /**
            * must be reflection
            */
            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFHelper helper,
                                       TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static float safe_PDF(float3 wo, float3 wi,
                                            BSDFHelper helper,
                                            TransportMode mode = TransportMode::Radiance);

            /**
             * eta must be corrected
             */
            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  Spectrum Fr, BSDFHelper helper,
                                                  TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                                 TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyRefl;
            }

        };

        class MicrofacetTransmission : public BxDF<MicrofacetTransmission> {
        public:
            using BxDF::BxDF;

            /**
             * must be transmission and eta must be corrected
             */
            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                                TransportMode mode = TransportMode::Radiance);

            /**
             * wo and wi must be not same hemisphere
             * and eta must be corrected
             */
            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFHelper helper,
                                       TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static float safe_PDF(float3 wo, float3 wi,
                                            BSDFHelper helper,
                                            TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  Spectrum Fr, BSDFHelper helper,
                                                  TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper data,
                                                 TransportMode mode = TransportMode::Radiance);

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyTrans;
            }

        };
    }
}