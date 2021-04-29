//
// Created by Zero on 2021/1/29.
//


#pragma once


#include "base.h"


namespace luminous {
    inline namespace render {

        class IdealDiffuse {
        private:
            float4 _R;
        public:
            IdealDiffuse() = default;

            IdealDiffuse(float4 R) : _R(R) {}

            XPU float4 eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _R * constant::invPi : make_float4(0.f);
            }

            XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                          BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
                return cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
            }
        };

    } // luminous::render
} //luminous