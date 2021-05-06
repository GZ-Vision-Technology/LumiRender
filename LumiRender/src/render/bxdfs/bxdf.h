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

            NDSC_XPU float4 base_color() const {
                LUMINOUS_VAR_DISPATCH(base_color);
            }

            NDSC_XPU float4 eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                LUMINOUS_VAR_DISPATCH(eval, wo, wi, mode);
            }

            NDSC_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                LUMINOUS_VAR_DISPATCH(PDF, wo, wi, mode, sample_flags);
            }

            NDSC_XPU float4 rho_hd(float3 wo, BufferView<const float> uc,
                                   BufferView<const float2> u2) const {
                if (wo.z == 0) {
                    return make_float4(0.f);
                }
                float4 r = make_float4(0.);
                DCHECK_EQ(uc.size(), u2.size());
                for (size_t i = 0; i < uc.size(); ++i) {
                    lstd::optional<BSDFSample> bs = sample_f(wo, uc[i], u2[i]);
                    if (bs) {
                        r += bs->f_val * Frame::abs_cos_theta(bs->wi) / bs->PDF;
                    }
                }
                return r / float(uc.size());
            }

            NDSC_XPU float4 rho_hh(BufferView<const float2> u1, BufferView<const float> uc,
                                   BufferView<const float2> u2) const {
                DCHECK_EQ(uc.size(), u1.size());
                DCHECK_EQ(u1.size(), u2.size());
                float4 r = make_float4(0.f);
                for (size_t i = 0; i < uc.size(); ++i) {
                    float3 wo = square_to_hemisphere(u1[i]);
                    if (wo.z == 0) {
                        continue;
                    }
                    float PDF_wo = uniform_hemisphere_PDF();
                    lstd::optional<BSDFSample> bs = sample_f(wo, uc[i], u2[i]);
                    if (bs) {
                        r = bs->f_val * Frame::abs_cos_theta(bs->wi) * Frame::abs_cos_theta(wo) / (PDF_wo * bs->PDF);
                    }
                }
                return r / (constant::Pi * uc.size());
            }

            NDSC_XPU lstd::optional<BSDFSample> sample_f(float3 wo, float uc, float2 u, TransportMode mode = TransportMode::Radiance,
                                                  BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                LUMINOUS_VAR_DISPATCH(sample_f, wo, uc, u, mode, sample_flags);
            }

            NDSC_XPU BxDFFlags flags() const {
                LUMINOUS_VAR_DISPATCH(flags);
            }
        };

    }
}