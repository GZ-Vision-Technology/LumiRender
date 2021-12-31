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
                float _weight{};
            public:
                using BxDF::BxDF;

                LM_XPU explicit Diffuse(float weight) : _weight(weight) {}

                ND_XPU_INLINE float weight() const {
                    return _weight;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                    float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    return _weight * helper.color() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
                }
            };

            class FakeSS : public BxDF<FakeSS> {
            private:
                float _weight{};
            public:
                ND_XPU_INLINE float weight() const {
                    return _weight;
                }

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float3 wh = wi + wo;
                    if (length_squared(wh) == 0.f) {
                        return {0.f};
                    }
                    wh = normalize(wh);
                    float cos_theta_d = dot(wi, wh);

                    float Fss90 = cos_theta_d * cos_theta_d * helper.roughness();
                    float Fo = schlick_weight(Frame::abs_cos_theta(wo)),
                    Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    float Fss = lerp(Fo, 1.f, Fss90) * lerp(Fi, 1.f, Fss90);
                    float ss = 1.25f * (Fss * (1 / (Frame::abs_cos_theta(wo) + Frame::abs_cos_theta(wi)) - .5f) + .5f);

                    return helper.color() * invPi * ss * _weight;
                }
            };

            class Retro : public BxDF<Retro> {
            private:
                float _weight{};
            public:
                ND_XPU_INLINE float weight() const {
                    return _weight;
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
                    float Rr = 2 * helper.roughness() * cos_theta_d * cos_theta_d;

                    return helper.color() * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1)) * _weight;
                }
            };

            class Sheen : public BxDF<Sheen> {
            private:
                float _weight{};
            public:
                ND_XPU_INLINE float weight() const {
                    return _weight;
                }
            };

            class Clearcoat : public BxDF<Clearcoat> {
            private:
                float _weight{};
                float _gloss{};
            public:
                ND_XPU_INLINE float weight() const {
                    return _weight;
                }
            };
        }
    }
}