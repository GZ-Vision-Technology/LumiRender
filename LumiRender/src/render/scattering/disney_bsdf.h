//
// Created by Zero on 27/12/2021.
//


#pragma once

#include "base.h"
#include "bsdf_data.h"
#include "bsdf_ty.h"
#include "diffuse_scatter.h"
#include "microfacet_scatter.h"
#include "specular_scatter.h"

namespace luminous {
    inline namespace render {
        namespace disney {
            class Diffuse : public BxDF<Diffuse> {
            public:
                using BxDF::BxDF;

                LM_XPU explicit Diffuse(Spectrum color) : BxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;
            };

            class FakeSS : public BxDF<FakeSS> {
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit FakeSS(Spectrum color) : BxDF(color, DiffRefl) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return 1.f;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

            };

            class Retro : public BxDF<Retro> {
            private:
                float _factor{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Retro(Spectrum color) : BxDF(color, DiffRefl) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return 1;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

            };

            class Sheen : public BxDF<Sheen> {
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Sheen(Spectrum color) : BxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return 1;
                }
            };

//            class Clearcoat : public BxDF<Clearcoat> {
//            private:
//                float _weight{};
//            public:
//                using BxDF::BxDF;
//            public:
//                LM_XPU explicit Clearcoat(float weight)
//                        : BxDF(_weight > 0),
//                          _weight(_weight) {}
//
//                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
//                                        TransportMode mode = TransportMode::Radiance) const;
//
//                /**
//                 * must be reflection
//                 */
//                LM_ND_XPU float PDF(float3 wo, float3 wi,
//                                    BSDFHelper helper,
//                                    TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
//                                         BSDFHelper helper,
//                                         TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
//                                              TransportMode mode = TransportMode::Radiance) const;
//
//                ND_XPU_INLINE float weight(BSDFHelper helper) const {
//                    return _weight;
//                }
//
//                LM_ND_XPU constexpr static BxDFFlags flags() {
//                    return BxDFFlags::GlossyRefl;
//                }
//            };

//            class DiffuseTransmission : public render::DiffuseTransmission {
//            public:
//                using render::DiffuseTransmission::DiffuseTransmission;
//
//                LM_ND_XPU float4 color(BSDFHelper helper) const {
//                    return helper.color() * helper.diff_trans();
//                }
//
//                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
//                                        TransportMode mode = TransportMode::Radiance) const {
//                    return _f(wo, wi, helper, helper.color(), mode);
//                }
//
//                LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
//                                             TransportMode mode = TransportMode::Radiance) const {
//                    return same_hemisphere(wo, wi) ? Spectrum{0.f} : eval(wo, wi, helper);
//                }
//
//                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
//                                              TransportMode mode = TransportMode::Radiance) const {
//                    return _sample_f(wo, uc, u, helper, helper.color() * helper.diff_trans(), mode);
//                }
//            };
//
//            class SpecularTransmission : public render::SpecularTransmission {
//            public:
//                using render::SpecularTransmission::SpecularTransmission;
//                LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
//                                               Spectrum Fr, BSDFHelper helper,
//                                               TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
//                                              TransportMode mode = TransportMode::Radiance) const;
//            };
//
//            class MicrofacetReflection : public render::MicrofacetReflection {
//            public:
//                using render::MicrofacetReflection::MicrofacetReflection;
//                /**
//                 * must be reflection and eta must be corrected
//                 */
//                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
//                                        TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
//                                             TransportMode mode = TransportMode::Radiance) const;
//
//                /**
//                 * eta must be corrected
//                 */
//                LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
//                                               Spectrum Fr, BSDFHelper helper,
//                                               TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
//                                              TransportMode mode = TransportMode::Radiance) const;
//            };
//
//            class MicrofacetTransmission : public render::MicrofacetTransmission {
//            public:
//                using render::MicrofacetTransmission::MicrofacetTransmission;
//                /**
//                 * must be reflection and eta must be corrected
//                 */
//                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
//                                        TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
//                                             TransportMode mode = TransportMode::Radiance) const;
//
//                /**
//                 * eta must be corrected
//                 */
//                LM_ND_XPU BSDFSample _sample_f(float3 wo, float uc, float2 u,
//                                               Spectrum Fr, BSDFHelper helper,
//                                               TransportMode mode = TransportMode::Radiance) const;
//
//                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
//                                              TransportMode mode = TransportMode::Radiance) const;
//            };
        }
    }
}