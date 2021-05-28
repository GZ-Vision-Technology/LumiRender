//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "diffuse.h"
#include "graphics/lstd/variant.h"
//#include "render/include/shader_data.h"
#include "graphics/optics/rgb.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class BxDF : public Variant<IdealDiffuse> {
            using Variant::Variant;
        public:
            GEN_BASE_NAME(BxDF)

            GEN_NAME_AND_TO_STRING_FUNC

            NDSC_XPU float4 base_color() const;

            NDSC_XPU Spectrum eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const;

            NDSC_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            NDSC_XPU Spectrum rho_hd(float3 wo, BufferView<const float> uc,
                                   BufferView<const float2> u2) const;

            NDSC_XPU Spectrum rho_hh(BufferView<const float2> u1, BufferView<const float> uc,
                                   BufferView<const float2> u2) const;

            NDSC_XPU lstd::optional<BSDFSample> sample_f(float3 wo, float uc, float2 u, TransportMode mode = TransportMode::Radiance,
                                                  BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            NDSC_XPU BxDFFlags flags() const;

            XPU void print() const;
        };

    }
}