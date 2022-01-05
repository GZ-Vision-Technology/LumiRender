//
// Created by Zero on 15/12/2021.
//


#pragma once

#include "base.h"
#include "base_libs/optics/rgb.h"
#include "bsdf_data.h"
#include "microfacet.h"

namespace luminous {
    inline namespace render {
        class DiffuseReflection : public BxDF<DiffuseReflection> {
        public:
            using BxDF::BxDF;

            LM_XPU explicit DiffuseReflection(Spectrum color) : BxDF(color, DiffRefl) {}

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const {
                return spectrum() * constant::invPi;
            }
        };

        class OrenNayar : public BxDF<OrenNayar> {
        public:
            using BxDF::BxDF;

            LM_XPU explicit OrenNayar(Spectrum color) : BxDF(color, DiffRefl) {}

            /**
             * fr(wi,wo) = R / PI * (A + B * max(0,cos(phi_i - phi_o)) * sin_alpha * tan_beta)
             * where A = 1 - sigma^2 / (2 * sigma^2 + 0.33)
             * 	     B = 0.45 * sigma^2 / (sigma^2 + 0.09)
             *   	 alpha = max(theta_i,theta_o)
             *		 beta = min(theta_i,theta_o)
             */
            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const;
        };

        class DiffuseTransmission : public BxDF<DiffuseTransmission> {
        protected:
            LM_ND_XPU Spectrum _f(float3 wo, float3 wi, BSDFHelper helper, Spectrum color,
                                  TransportMode mode = TransportMode::Radiance) const {
                return color * constant::invPi;
            }

            LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u, BSDFHelper helper, Spectrum color,
                                           TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z > 0 ? -wi.z : wi.z;
                float PDF_val = safe_PDF(wo, wi, helper, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _f(wo, wi, helper, color, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

        public:
            using BxDF::BxDF;

            LM_XPU explicit DiffuseTransmission(Spectrum color) : BxDF(color, DiffTrans) {}

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const {
                return _f(wo, wi, helper, spectrum(), mode);
            }

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? Spectrum{0.f} : eval(wo, wi, helper);
            }

            LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                     BSDFHelper helper,
                                     TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? 0.f : cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const {
                return _sample_f(wo, uc, u, helper, helper.color(), mode);
            }
        };
    }
}