//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "diffuse.h"
#include "graphics/lstd/variant.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class BxDF : public Variant<IdealDiffuse> {
            using Variant::Variant;
        public:
            GEN_BASE_NAME(BxDF)

            GEN_NAME_AND_TO_STRING_FUNC

            XPU float4 eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                LUMINOUS_VAR_DISPATCH(eval, wo, wi, mode);
            }

            XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                LUMINOUS_VAR_DISPATCH(PDF, wo, wi, mode, sample_flags);
            }

            XPU lstd::optional<BSDFSample> sample(float3 wo, float2 u, TransportMode mode = TransportMode::Radiance,
                                                  BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                LUMINOUS_VAR_DISPATCH(sample, wo, u, mode, sample_flags);
            }

            NDSC_XPU BxDFFlags flags() const {
                LUMINOUS_VAR_DISPATCH(flags);
            }
        };

    }
}