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

        Spectrum BxDF::rho_hd(float3 wo, BufferView<const float> uc, BufferView<const float2> u2) const {
            if (wo.z == 0) {
                return make_float4(0.f);
            }
            Spectrum r = make_float4(0.);
            DCHECK_EQ(uc.size(), u2.size());
            for (size_t i = 0; i < uc.size(); ++i) {
                auto bs = sample_f(wo, uc[i], u2[i]);
                if (bs.valid()) {
                    r += bs.f_val * Frame::abs_cos_theta(bs.wi) / bs.PDF;
                }
            }
            return r / float(uc.size());
        }

        Spectrum BxDF::rho_hh(BufferView<const float2> u1, BufferView<const float> uc, BufferView<const float2> u2) const {
            DCHECK_EQ(uc.size(), u1.size());
            DCHECK_EQ(u1.size(), u2.size());
            Spectrum r = make_float4(0.f);
            for (size_t i = 0; i < uc.size(); ++i) {
                float3 wo = square_to_hemisphere(u1[i]);
                if (wo.z == 0) {
                    continue;
                }
                float PDF_wo = uniform_hemisphere_PDF();
                auto bs = sample_f(wo, uc[i], u2[i]);
                if (bs.valid()) {
                    r += bs.f_val * Frame::abs_cos_theta(bs.wi) * Frame::abs_cos_theta(wo) / (PDF_wo * bs.PDF);
                }
            }
            return r / (constant::Pi * uc.size());
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