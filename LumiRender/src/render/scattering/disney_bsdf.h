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
            private:
                float _factor{};
            public:
                using BxDF::BxDF;

                LM_XPU explicit Diffuse(float factor) : BxDF(factor > 0), _factor(factor) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor * luminance(helper.color());
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU constexpr static BxDFFlags flags() {
                    return BxDFFlags::DiffRefl;
                }

            };

            class FakeSS : public BxDF<FakeSS> {
            private:
                float _factor{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit FakeSS(float factor) : BxDF(factor > 0), _factor(factor) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor * luminance(helper.color());
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU constexpr static BxDFFlags flags() {
                    return BxDFFlags::DiffRefl;
                }
            };

            class Retro : public BxDF<Retro> {
            private:
                float _factor{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Retro(float factor) : BxDF(factor > 0), _factor(factor) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor * luminance(helper.color());
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU constexpr static BxDFFlags flags() {
                    return BxDFFlags::DiffRefl;
                }
            };

            class Sheen : public BxDF<Sheen> {
            private:
                float _factor{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Sheen(float factor) : BxDF(factor > 0), _factor(factor) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor;
                }

                LM_ND_XPU constexpr static BxDFFlags flags() {
                    return BxDFFlags::DiffRefl;
                }
            };

            class Clearcoat : public BxDF<Clearcoat> {
            private:
                float _weight{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Clearcoat(float weight)
                        : BxDF(_weight > 0),
                          _weight(_weight) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
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

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _weight;
                }

                LM_ND_XPU constexpr static BxDFFlags flags() {
                    return BxDFFlags::GlossyRefl;
                }
            };

            class DiffuseTransmission : public render::DiffuseTransmission {
            public:
                LM_XPU DiffuseTransmission() = default;

                LM_ND_XPU float4 color(BSDFHelper helper) const {
                    return helper.color() * helper.diff_trans();
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    return _f(wo, wi, helper, helper.color(), mode);
                }

                LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                             TransportMode mode = TransportMode::Radiance) const {
                    return same_hemisphere(wo, wi) ? Spectrum{0.f} : eval(wo, wi, helper);
                }

                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                              TransportMode mode = TransportMode::Radiance) const {
                    return _sample_f(wo, uc, u, helper, helper.color() * helper.diff_trans(), mode);
                }
            };

            class SpecularTransmission : public render::SpecularTransmission {
            public:

            };
        }
    }
}