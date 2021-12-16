//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base.h"
#include "fresnel.h"
#include "microfacet.h"

namespace luminous {
    inline namespace render {
        class DielectricBxDF {
        private:
            float4 Kr;
            float4 Kt;
            float _eta{0.f};
            Microfacet<> _microfacet{};
        public:
            LM_XPU DielectricBxDF() = default;

            LM_ND_XPU float4 color() const {
                return Kr;
            }

            LM_XPU DielectricBxDF(float4 kr, float4 kt, float eta)
                    : Kr(kr), Kt(kt), _eta(eta) {}

            LM_XPU DielectricBxDF(float4 kr, float4 kt, float eta, Microfacet<> distribution)
                    : Kr(kr), Kt(kt), _eta(eta), _microfacet(distribution) {}

            LM_XPU DielectricBxDF(float4 kr, float4 kt, float eta, float rx, float ry)
                    : Kr(kr), Kt(kt), _eta(eta), _microfacet(Microfacet<>(rx, ry)) {}

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                                BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                          TransportMode mode = TransportMode::Radiance,
                                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU BxDFFlags flags() const;

            LM_XPU void print() const {
                printf("DielectricBxDF Kr(%f,%f,%f,%f)", Kr.x, Kr.y, Kr.z, Kr.w);
            }

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("DielectricBxDF Kr : %s", Kr.to_string().c_str());
                            })
        };
    }
}