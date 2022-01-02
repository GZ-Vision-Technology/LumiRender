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
                    return _factor;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                    float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    return _factor * helper.color() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
                }

                ND_XPU_INLINE static Diffuse create(float factor) {
                    return Diffuse(factor);
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
                    return _factor;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float3 wh = wi + wo;
                    if (length_squared(wh) == 0.f) {
                        return {0.f};
                    }
                    wh = normalize(wh);
                    float cos_theta_d = dot(wi, wh);

                    float Fss90 = sqr(cos_theta_d) * helper.roughness();
                    float Fo = schlick_weight(Frame::abs_cos_theta(wo)),
                            Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    float Fss = lerp(Fo, 1.f, Fss90) * lerp(Fi, 1.f, Fss90);
                    float ss = 1.25f * (Fss * (1 / (Frame::abs_cos_theta(wo) + Frame::abs_cos_theta(wi)) - .5f) + .5f);

                    return helper.color() * invPi * ss * _factor;
                }

                ND_XPU_INLINE static FakeSS create(float factor) {
                    return FakeSS(factor);
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
                    return _factor;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float3 wh = wi + wo;
                    if (length_squared(wh) == 0.f) {
                        return {0.f};
                    }
                    wh = normalize(wh);
                    float cos_theta_d = dot(wi, wh);

                    float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                    float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    float Rr = 2 * helper.roughness() * sqr(cos_theta_d);

                    return helper.color() * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1)) * _factor;
                }
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