//
// Created by Zero on 2021/5/13.
//

#include "bxdf.h"

namespace luminous {
    inline namespace render {

        float4 BxDF::base_color() const {
            LUMINOUS_VAR_DISPATCH(base_color);
        }

        Spectrum BxDF::eval(float3 wo, float3 wi, TransportMode mode) const {
            LUMINOUS_VAR_DISPATCH(eval, wo, wi, mode);
        }

        float BxDF::PDF(float3 wo, float3 wi, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            LUMINOUS_VAR_DISPATCH(PDF, wo, wi, mode, sample_flags);
        }

        BSDFSample BxDF::sample_f(float3 wo, float uc, float2 u, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            LUMINOUS_VAR_DISPATCH(sample_f, wo, uc, u, mode, sample_flags);
        }

        BxDFFlags BxDF::flags() const {
            LUMINOUS_VAR_DISPATCH(flags);
        }

        void BxDF::print() const {
            LUMINOUS_VAR_DISPATCH(print);
        }
    } // luminous::render
} // luminous