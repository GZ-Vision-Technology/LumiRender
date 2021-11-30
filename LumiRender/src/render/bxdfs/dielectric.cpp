//
// Created by Zero on 29/11/2021.
//

#include "dielectric.h"

namespace luminous {
    inline namespace render {

        Spectrum DielectricBxDF::eval(float3 wo, float3 wi, TransportMode mode) const {
            if (_distribution.effectively_smooth()) {
                return 0.f;
            }

            return {};
        }

        float DielectricBxDF::PDF(float3 wo, float3 wi, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            if (_distribution.effectively_smooth()) {
                return 0.f;
            }
            return 0;
        }

        lstd::optional<BSDFSample> DielectricBxDF::sample_f(float3 wo, float uc, float2 u, TransportMode mode,
                                                            BxDFReflTransFlags sample_flags) const {
            return lstd::optional<BSDFSample>();
        }

        BxDFFlags DielectricBxDF::flags() const {
            if (_eta == 1)
                return BxDFFlags::Transmission;
            else
                return BxDFFlags::Reflection | BxDFFlags::Transmission |
                       (_distribution.effectively_smooth() ? BxDFFlags::Specular : BxDFFlags::Glossy);
        }
    }
}