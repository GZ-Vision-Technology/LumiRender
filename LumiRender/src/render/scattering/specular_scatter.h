//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        class SpecularReflection : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  float Fr, BSDFData data,
                                                  Fresnel fresnel,
                                                  Microfacet microfacet = {},
                                                  TransportMode mode = TransportMode::Radiance) {
                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                Spectrum val = Fr * Spectrum(data.color()) / Frame::abs_cos_theta(wi);
                float PDF = 1.f;
                return {val, wi, PDF, BxDFFlags::SpecRefl, data.eta()};
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel,
                                                 Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {

                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                data.correct_eta(Frame::cos_theta(wo), fresnel.type());
                auto Fr = fresnel.eval(Frame::abs_cos_theta(wo), data)[0];
                return _sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class SpecularTransmission : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static BSDFSample _sample_f(float3 wo, float uc, float2 u,
                                                  float Fr, BSDFData data,
                                                  Fresnel fresnel,
                                                  Microfacet microfacet = {},
                                                  TransportMode mode = TransportMode::Radiance) {
                float3 wi{};
                float3 n = make_float3(0, 0, 1);
                bool valid = refract(wo, face_forward(n, wo), data.eta(), &wi);
                if (!valid) {
                    return {};
                }
                auto ft = (1.f - Fr) / Frame::abs_cos_theta(wi);
                float factor = cal_factor(mode, data.eta());
                Spectrum val = ft * data.color() * factor;
                return {val, wi, 1, SpecTrans, data.eta()};
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel,
                                                 Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float3 wi{};
                data.correct_eta(Frame::cos_theta(wo), fresnel.type());
                float3 n = make_float3(0, 0, 1);
                bool valid = refract(wo, face_forward(n, wo), data.eta(), &wi);
                if (!valid) {
                    return {};
                }
                auto Fr = fresnel.eval(Frame::abs_cos_theta(wo), data)[0];
                return _sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecTrans;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class SpecularFresnel : public BxDF {
        public:
            using BxDF::BxDF;

            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, BSDFData data,
                                           Fresnel fresnel,
                                           Microfacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       BSDFData data,
                                       Fresnel fresnel,
                                       Microfacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFData data,
                                                 Fresnel fresnel,
                                                 Microfacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                data.correct_eta(cos_theta_o, fresnel.type());
                float Fr = fresnel.eval(Frame::abs_cos_theta(wo), data)[0];
                BSDFSample ret;
                if (uc < Fr) {
                    ret = SpecularReflection::_sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
                    ret.PDF *= Fr;
                } else {
                    ret = SpecularTransmission::_sample_f(wo, uc, u, Fr, data, fresnel, microfacet, mode);
                    ret.PDF *= 1 - Fr;
                }
                return ret;
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecTrans | BxDFFlags::SpecRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };
    }
}