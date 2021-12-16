//
// Created by Zero on 2021/1/29.
//


#pragma once


#include "base.h"
#include "base_libs/optics/rgb.h"
#include "core/logging.h"

namespace luminous {
    inline namespace render {

        class IdealDiffuse {
        private:
            float4 R = make_float4(-1.f);

            LM_ND_XPU Spectrum _eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                return Spectrum{R * constant::invPi};
            }

        public:
            LM_XPU IdealDiffuse() = default;

            LM_XPU explicit IdealDiffuse(float4 R) : R(R) {}

            LM_ND_XPU float4 color() const {
                return R;
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _eval(wo, wi) : Spectrum{make_float4(0.f)};
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                                BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                                          TransportMode mode = TransportMode::Radiance,
                                                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
                    return {};
                }
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = PDF(wo, wi, mode, sample_flags);
                if (PDF_val == 0) {
                    return {};
                }
                Spectrum f = _eval(wo, wi, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            LM_ND_XPU BxDFFlags flags() const {
                return BxDFFlags::DiffRefl;
            }

            LM_ND_XPU bool match_flags(BxDFFlags bxdf_flags) const {
                return (flags() & bxdf_flags) == flags();
            }

            LM_XPU void print() const {
                printf("ideal diffuse r(%f,%f,%f,%f)", R.x, R.y, R.z, R.w);
            }

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("IdealDiffuse R : %s", R.to_string().c_str());
            })
        };
    } // luminous::render
} //luminous