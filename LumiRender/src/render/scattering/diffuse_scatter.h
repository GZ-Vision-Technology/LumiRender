//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"
#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {
        class DiffuseReflection {
        public:
            LM_XPU DiffuseReflection() = default;

            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, Spectrum R,
                                     TransportMode mode = TransportMode::Radiance) const {
                return Spectrum{R * constant::invPi};
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, Spectrum R,
                                    TFresnel fresnel = {},
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _eval(wo, wi, R) : Spectrum{0.f};
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, Spectrum R, TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, fresnel, microfacet, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, R, fresnel, microfacet, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            LM_ND_XPU BxDFFlags flags() const {
                return BxDFFlags::DiffRefl;
            }

            LM_ND_XPU bool match_flags(BxDFFlags bxdf_flags) const {
                return (flags() & bxdf_flags) == flags();
            }
        };

        class DiffuseTransmission {
        public:
            LM_XPU DiffuseTransmission() = default;

            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, Spectrum T,
                                     TransportMode mode = TransportMode::Radiance) const {
                return T * constant::invPi;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, Spectrum T,
                                    TFresnel fresnel = {},
                                    TMicrofacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? Spectrum{0.f} : _eval(wo, wi, T);
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                TFresnel fresnel = {},
                                TMicrofacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? 0.f : cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u, Spectrum R, TFresnel fresnel,
                                          TMicrofacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z > 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, fresnel, microfacet, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, R, fresnel, microfacet, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }
        };
    }
}