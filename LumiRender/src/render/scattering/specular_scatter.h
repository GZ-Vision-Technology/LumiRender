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

            ND_XPU_INLINE float weight(BSDFHelper helper, Spectrum Fr) const {
                return ColoredBxDF::weight(helper, Fr) * Fr.luminance();
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

            ND_XPU_INLINE float weight(BSDFHelper helper, Spectrum Fr) const {
                return luminance((spectrum() * (Spectrum(1.f) - Fr)));
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

        extern LM_XPU float fresnel_moment1(float eta);
        extern LM_XPU float fresnel_moment2(float eta);

        class NormalizedFresnelBxDF : public BxDF<NormalizedFresnelBxDF> {
        public:
            NormalizedFresnelBxDF(float eta);

            BSDFSample sample_f(float3 wo, float uc, float2 u, const BSDFHelper &data, TransportMode mode = TransportMode::Radiance) const;

            Spectrum safe_eval(float3 wo, float3 wi, const BSDFHelper &data, TransportMode mode = TransportMode::Radiance) const;

            float safe_PDF(float3 wo, float3 wi, const BSDFHelper &data, TransportMode mode = TransportMode::Radiance) const;

            Spectrum spectrum() const {
                return {};
            }

        private:
            float _inv_c_mul_pi;
            float _eta;
        };
    }
}