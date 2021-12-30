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

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper data,
                                        TransportMode mode = TransportMode::Radiance) const {
                    float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                    float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                    return _weight * data.color() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
                }
            };

            class FakeSS : public BxDF<FakeSS> {
            private:
                float _roughness{};
                float _weight{};
            };

            class Retro : public BxDF<Retro> {
            private:
                float _roughness{};
                float _weight{};
            };

            class Sheen : public BxDF<Sheen> {
            private:
                float _weight{};
            };

            class Clearcoat : public BxDF<Clearcoat> {
            private:
                float _weight{};
                float _gloss{};
            };
        }
    }
}