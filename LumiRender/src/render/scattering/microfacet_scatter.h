//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"

namespace luminous {
    inline namespace render {

        class MicrofacetReflection : public ColoredBxDF<MicrofacetReflection> {
        protected:
            Microfacet _microfacet{};

            LM_ND_XPU Spectrum _f(float3 wo, float3 wi, BSDFHelper helper, Spectrum color,
                                  TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample _sample_f_color(float3 wo, float uc, float2 u,
                                                 Spectrum Fr, BSDFHelper helper, Spectrum color,
                                                 TransportMode mode = TransportMode::Radiance) const;

        public:
            using ColoredBxDF::ColoredBxDF;

            LM_XPU explicit MicrofacetReflection(Spectrum color, Microfacet microfacet)
                    : ColoredBxDF(color, GlossyRefl), _microfacet(microfacet) {}

            LM_XPU explicit MicrofacetReflection(Spectrum color, float alpha_x, float alpha_y, MicrofacetType type)
                    : ColoredBxDF(color, GlossyRefl), _microfacet(alpha_x, alpha_y, type) {}

            /**
             * must be reflection and eta must be corrected
             */
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const;

            /**
             * must be reflection
             */
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BSDFHelper helper,
                                TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                     BSDFHelper helper,
                                     TransportMode mode = TransportMode::Radiance) const;

            /**
             * eta must be corrected
             */
            LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                           Spectrum Fr, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;
        };

        class MicrofacetTransmission : public ColoredBxDF<MicrofacetTransmission> {
        protected:
            Microfacet _microfacet{};

            LM_ND_XPU Spectrum _f(float3 wo, float3 wi, BSDFHelper helper, Spectrum color,
                                  TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample _sample_f_color(float3 wo, float uc, float2 u,
                                                 Spectrum Fr, BSDFHelper helper, Spectrum color,
                                                 TransportMode mode = TransportMode::Radiance) const;

        public:
            using ColoredBxDF::ColoredBxDF;

            LM_XPU explicit MicrofacetTransmission(Spectrum color, Microfacet microfacet)
                    : ColoredBxDF(color, GlossyRefl), _microfacet(microfacet) {}

            LM_XPU explicit MicrofacetTransmission(Spectrum color, float alpha_x, float alpha_y, MicrofacetType type)
                    : ColoredBxDF(color, GlossyRefl), _microfacet(alpha_x, alpha_y, type) {}
            /**
             * must be transmission and eta must be corrected
             */
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const;

            /**
             * wo and wi must be not same hemisphere
             * and eta must be corrected
             */
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BSDFHelper helper,
                                TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                     BSDFHelper helper,
                                     TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                           Spectrum Fr, BSDFHelper helper,
                                           TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;
        };
    }
}