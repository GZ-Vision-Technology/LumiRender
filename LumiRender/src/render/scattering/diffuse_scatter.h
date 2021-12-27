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
        class DiffuseReflection : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, BSDFData data,
                                     TransportMode mode = TransportMode::Radiance) const {
                return Spectrum{data.color() * constant::invPi};
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                    Microfacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _eval(wo, wi, data) : Spectrum{0.f};
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BSDFData data,
                                Microfacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0.f;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                          Microfacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, data, microfacet, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, data, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::DiffRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class OrenNayar : public BxDF {
        public:
            using BxDF::BxDF;

            /**
             * fr(wi,wo) = R / PI * (A + B * max(0,cos(phi_i - phi_o)) * sin_alpha * tan_beta)
             * where A = 1 - sigma^2 / (2 * sigma^2 + 0.33)
             * 	     B = 0.45 * sigma^2 / (sigma^2 + 0.09)
             *   	 alpha = max(theta_i,theta_o)
             *		 beta = min(theta_i,theta_o)
             */
            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, BSDFData data,
                                     TransportMode mode = TransportMode::Radiance) const {
                float sin_theta_i = Frame::sin_theta(wi);
                float sin_theta_o = Frame::sin_theta(wo);

                float max_cos = 0;

                if (sin_theta_i > 1e-4 && sin_theta_o > 1e-4) {
                    float sinPhiI = Frame::sin_phi(wi), cosPhiI = Frame::cos_phi(wi);
                    float sinPhiO = Frame::sin_phi(wo), cosPhiO = Frame::cos_phi(wo);
                    float d_cos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
                    max_cos = std::max(0.f, d_cos);
                }

                float sin_alpha, tan_beta;
                bool condition = Frame::abs_cos_theta(wi) > Frame::abs_cos_theta(wo);
                sin_alpha = condition ? sin_theta_o : sin_theta_i;
                tan_beta = condition ?
                           sin_theta_i / Frame::abs_cos_theta(wi) :
                           sin_theta_o / Frame::abs_cos_theta(wo);
                float2 AB = data.AB();
                float factor = (AB.x + AB.y * max_cos * sin_alpha * tan_beta);
                return data.color() * invPi * factor;
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                    Microfacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _eval(wo, wi, data) : Spectrum{0.f};
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BSDFData data,
                                Microfacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0.f;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                          Microfacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, data, microfacet, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, data, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::DiffRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class DiffuseTransmission : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, BSDFData data,
                                     TransportMode mode = TransportMode::Radiance) const {
                return Spectrum{data.color() * constant::invPi};
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                    Microfacet microfacet = {},
                                    TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? Spectrum{0.f} : _eval(wo, wi, data);
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BSDFData data,
                                Microfacet microfacet = {},
                                TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? 0.f : cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                          Microfacet microfacet = {},
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z > 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, data, microfacet, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, data, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::DiffTrans;
            }

            GEN_MATCH_FLAGS_FUNC
        };
    }
}