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


        class MicrofacetFresnel : public BxDF<MicrofacetFresnel> {
        public:
            using BxDF::BxDF;

            LM_ND_XPU static Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                                TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                if (same_hemisphere(wi, wo)) {
                    return MicrofacetReflection::eval(wo, wi, helper, mode);
                }
                return MicrofacetTransmission::eval(wo, wi, helper, mode);
            }

            LM_ND_XPU static float safe_PDF(float3 wo, float3 wi,
                                            BSDFHelper helper,
                                            TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                if (same_hemisphere(wi, wo)) {
                    return MicrofacetReflection::PDF(wo, wi, helper, mode);
                }
                return MicrofacetTransmission::PDF(wo, wi, helper, mode);
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                float Fr = helper.eval_fresnel(Frame::abs_cos_theta(wo))[0];
                BSDFSample ret;
                if (uc < Fr) {
                    // sample reflection
                    ret = MicrofacetReflection::_sample_f(wo, uc, u, Fr, helper, mode);
                    ret.PDF *= Fr;
                } else {
                    // sample transmission
                    ret = MicrofacetTransmission::_sample_f(wo, uc, u, Fr, helper, mode);
                    ret.PDF *= 1 - Fr;
                }
                return ret;
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::GlossyTrans | BxDFFlags::GlossyRefl;
            }
        };
    }
}