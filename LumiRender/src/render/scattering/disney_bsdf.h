//
// Created by Zero on 27/12/2021.
//


#pragma once

#include "base.h"
#include "bsdf_data.h"
#include "bsdf_ty.h"

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
        }
    }
}