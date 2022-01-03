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
            };

            class Sheen : public BxDF<Sheen> {
            private:
                float _factor{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Sheen(float factor) : BxDF(factor > 0), _factor(factor) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor;
                }
            };

            class Clearcoat : public BxDF<Clearcoat> {
            private:
                float _factor{};
                float _gloss{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Clearcoat(float factor, float gloss)
                        : BxDF(factor > 0),
                          _factor(factor), _gloss(gloss) {}

                ND_XPU_INLINE float weight(BSDFHelper helper) const {
                    return _factor;
                }
            };
        }
    }
}