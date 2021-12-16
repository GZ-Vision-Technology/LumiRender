//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "diffuse.h"
#include "base_libs/lstd/variant.h"
#include "base_libs/optics/rgb.h"
#include "dielectric.h"
#include "microfacet.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class BxDF : public Variant<IdealDiffuse, DielectricBxDF> {
            using Variant::Variant;
        public:
            GEN_BASE_NAME(BxDF)

            GEN_TO_STRING_FUNC

            LM_ND_XPU float4 base_color() const;

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                                BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, TransportMode mode = TransportMode::Radiance,
                                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU BxDFFlags flags() const;

            LM_XPU void print() const;
        };

    }
}