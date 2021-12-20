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

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, TData data,
                                           TFresnel fresnel,
                                           TMicrofacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       TFresnel fresnel = {},
                                       TMicrofacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            /**
             * @param wo
             * @param data
             * @param Fr fresnel function value
             * @param eta eta must be corrected
             * @return
             */
            template<typename TData, typename eta_type>
            LM_ND_XPU static BSDFSample sample_f(float3 wo, TData data, eta_type Fr, eta_type eta) {
                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                Spectrum val = Fr * Spectrum(data.color) / Frame::abs_cos_theta(wi);
                float PDF = 1.f;
                return {val, wi, PDF, BxDFFlags::SpecRefl, eta};
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, TData data,
                                                 TFresnel fresnel,
                                                 TMicrofacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {

                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                fresnel.correct_eta(Frame::cos_theta(wo));
                auto Fr = fresnel.eval(Frame::abs_cos_theta(wo));
                return sample_f(wo, data, Fr, fresnel.eta);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecRefl;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class SpecularTransmission : public BxDF {
        public:
            using BxDF::BxDF;

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, TData data,
                                           TFresnel fresnel,
                                           TMicrofacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       TFresnel fresnel = {},
                                       TMicrofacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            template<typename TData, typename eta_type>
            LM_ND_XPU static BSDFSample sample_f(float3 wo, float3 wi, TData data, eta_type Fr, eta_type eta,
                                                 TransportMode mode = TransportMode::Radiance) {
                auto ft = (eta_type(1) - Fr) / Frame::abs_cos_theta(wi);
                eta_type factor = cal_factor(mode, eta);
                Spectrum val = ft * data.color * factor;
                return {val, wi, 1, SpecTrans, eta};
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, TData data,
                                                 TFresnel fresnel,
                                                 TMicrofacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float3 wi{};
                fresnel.correct_eta(Frame::cos_theta(wo));
                float3 n = make_float3(0,0,1);
                bool valid = refract(wo, face_forward(n, wo), fresnel.eta, &wi);
                if (!valid) {
                    return {};
                }
                auto Fr = fresnel.eval(Frame::abs_cos_theta(wo));
                return sample_f(wo, wi, data, Fr, fresnel.eta, mode);
            }

            LM_ND_XPU constexpr static BxDFFlags flags() {
                return BxDFFlags::SpecTrans;
            }

            GEN_MATCH_FLAGS_FUNC
        };

        class SpecularFresnel : public BxDF {
        public:
            using BxDF::BxDF;

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static Spectrum eval(float3 wo, float3 wi, TData data,
                                           TFresnel fresnel,
                                           TMicrofacet microfacet = {},
                                           TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            template<typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static float PDF(float3 wo, float3 wi,
                                       TFresnel fresnel = {},
                                       TMicrofacet microfacet = {},
                                       TransportMode mode = TransportMode::Radiance) {
                return 0.f;
            }

            template<typename TData, typename TFresnel, typename TMicrofacet>
            LM_ND_XPU static BSDFSample sample_f(float3 wo, float uc, float2 u, TData data,
                                                 TFresnel fresnel,
                                                 TMicrofacet microfacet = {},
                                                 TransportMode mode = TransportMode::Radiance) {
                float cos_theta_o = Frame::cos_theta(wo);
                fresnel.correct_eta(cos_theta_o);
                float Fr = fresnel.eval(Frame::abs_cos_theta(wo));
                BSDFSample ret;
                if (uc < Fr) {
                    ret = SpecularReflection::sample_f(wo, data, Fr, fresnel.eta);
                    ret.PDF = Fr;
                } else {
                    float3 wi{};
                    float3 n = make_float3(0, 0, 1);
                    bool valid = refract(wo, face_forward(n, wo), fresnel.eta, &wi);
                    if (!valid) {
                        return {};
                    }
                    ret = SpecularTransmission::sample_f(wo, wi, data, Fr, fresnel.eta, mode);
                    ret.PDF = 1 - Fr;
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