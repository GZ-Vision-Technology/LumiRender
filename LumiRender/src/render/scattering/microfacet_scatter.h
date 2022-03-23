//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"

namespace luminous {

    // Forward declaration for Neubelt cloth BRDF
    namespace render {
        class ImageTexture;
    };

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

            LM_ND_XPU float weight(BSDFHelper helper, Spectrum Fr) const;

            LM_XPU explicit MicrofacetReflection(Spectrum color, Microfacet microfacet)
                    : ColoredBxDF(color, flags_by_alpha(microfacet.max_alpha()) | Reflection),
                      _microfacet(microfacet) {}

            LM_XPU explicit MicrofacetReflection(Spectrum color, float alpha_x, float alpha_y, MicrofacetType type)
                    : ColoredBxDF(color, flags_by_alpha(alpha_x, alpha_y) | Reflection),
                      _microfacet(alpha_x, alpha_y, type) {}

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
                    : ColoredBxDF(color, flags_by_alpha(microfacet.max_alpha()) | Transmission),
                      _microfacet(microfacet) {}

            LM_XPU explicit MicrofacetTransmission(Spectrum color, float alpha_x, float alpha_y, MicrofacetType type)
                    : ColoredBxDF(color, flags_by_alpha(alpha_x, alpha_y) | Transmission),
                      _microfacet(alpha_x, alpha_y, type) {}

            ND_XPU_INLINE float weight(BSDFHelper helper, Spectrum Fr) const {
                return luminance(spectrum() * (Spectrum(1.f) - Fr));
            }

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

        /**
         * diffuse and clearcoat blend
         *
         * specular is
         *
         *                                  D(wh) F(wo)
         * fr(p, wo, wi) = -----------------------------------------------
         *                   4 * dot(wh, wi) * max(dot(n,wi), dot(n,wo))
         *
         *
         * Fr(cos_theta) = R +(1 - R)(1 - cos_theta)^5
         *
         * diffuse is
         *
         *                  28 * Rd
         * fr(p, wo, wi) = --------- (1 - Rs) (1 - (1 - dot(n, wi)/2)^5) (1 - (1 - dot(n, wo)/2)^5)
         *                  23 * pi
         *
         */
        class MicrofacetFresnel : public ColoredBxDF<MicrofacetFresnel> {
        protected:
            Microfacet _microfacet{};
            Spectrum _spec{};
        public:
            using ColoredBxDF::ColoredBxDF;

            LM_XPU explicit MicrofacetFresnel(Spectrum color, Spectrum spec, Microfacet microfacet)
                    : ColoredBxDF(color, flags_by_alpha(microfacet.max_alpha()) | Reflection),
                      _spec(spec),
                      _microfacet(microfacet) {}

            LM_XPU explicit MicrofacetFresnel(Spectrum color, Spectrum spec, float alpha_x, float alpha_y,
                                              MicrofacetType type)
                    : ColoredBxDF(color, flags_by_alpha(alpha_x, alpha_y) | Reflection),
                      _spec(spec),
                      _microfacet(alpha_x, alpha_y, type) {}

            LM_ND_XPU Spectrum schlick_fresnel(float cos_theta, BSDFHelper helper) const;

            LM_ND_XPU Spectrum eval_diffuse(float3 wo, float3 wi, BSDFHelper helper,
                                            TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF_diffuse(float3 wo, float3 wi,
                                        BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU Spectrum eval_specular(float3 wo, float3 wi, BSDFHelper helper,
                                            TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF_specular(float3 wo, float3 wi,
                                        BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

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

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;
        };

        class ClothMicrofacetFresnel: public ColoredBxDF<ClothMicrofacetFresnel> {
        public:
            LM_XPU ClothMicrofacetFresnel(Spectrum base_color, Spectrum spec_tint, float alpha,
                                          const ImageTexture *spec_albedo, const ImageTexture *spec_albedo_avg)
                : ColoredBxDF(base_color, GlossyRefl), _microfacet(alpha, NeubeltCloth), _spec_tint(spec_tint), _spec_albedo(spec_albedo), _spec_albedo_avg(spec_albedo_avg) {}

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

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const;

            protected:
                Spectrum eval_specluar(float3 wo, float3 wi, BSDFHelper data, TransportMode mode) const;
                Spectrum eval_diffuse(float3 wo, float3 wi, BSDFHelper data, TransportMode mode) const;

                Microfacet _microfacet;
                Spectrum _spec_tint;
                const ImageTexture *_spec_albedo, *_spec_albedo_avg;
        };
    }
}