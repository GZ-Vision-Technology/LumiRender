//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        template<typename Fresnel>
        class SpecularReflection {
        private:
            float3 _r;
            Fresnel _fresnel;
        public:
            LM_XPU SpecularReflection(float3 r, Fresnel fresnel)
                : _r(r), _fresnel(fresnel) {}

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi) const {
                return 0.f;
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi) const {
                return 0.f;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float2 u,
                                    TransportMode mode = TransportMode::Radiance) const {
                float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                Spectrum val = _fresnel.eval(Frame::cos_theta(wi)) * _r / Frame ::abs_cos_theta(wi);
                float PDF = 1.f;
                return {val, wi, PDF, BxDFFlags::SpecRefl};
            }
        };
    }
}