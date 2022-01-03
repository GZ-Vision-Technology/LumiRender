//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "flags.h"
#include "microfacet.h"
#include "bsdf_data.h"

namespace luminous {
    inline namespace render {

        struct BSDFSample {
            Spectrum f_val{};
            float3 wi{};
            float PDF{-1.f};
            BxDFFlags flags{};
            float eta{1.f};

            LM_ND_XPU BSDFSample() = default;

            LM_ND_XPU BSDFSample(const Spectrum &val, float3 wi_, float PDF_, BxDFFlags flags_, float eta_ = 1)
                    : f_val(val), wi(wi_), PDF(PDF_), flags(flags_), eta(eta_) {
                CHECK_UNIT_VEC(wi_)
            }

            LM_ND_XPU bool valid() const {
                return PDF >= 0.f;
            }

            LM_ND_XPU bool is_non_specular() const {
                return luminous::is_non_specular(flags);
            }

            LM_ND_XPU bool is_reflective() const {
                return luminous::is_reflective(flags);
            }

            LM_ND_XPU bool is_transmissive() const {
                return luminous::is_transmissive(flags);
            }

            LM_ND_XPU bool is_diffuse() const {
                return luminous::is_diffuse(flags);
            }

            LM_ND_XPU bool is_glossy() const {
                return luminous::is_glossy(flags);
            }

            LM_ND_XPU bool is_specular() const {
                return luminous::is_specular(flags);
            }
        };

        template<typename T>
        struct BxDF {
        public:
            const bool valid{};
        public:
            LM_XPU explicit BxDF(bool valid = true) : valid(valid) {}

            ND_XPU_INLINE float weight(BSDFHelper helper) const {
                return 1.f;
            }

            ND_XPU_INLINE float safe_PDF(float3 wo, float3 wi, BSDFHelper data,
                                         TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0.f;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper data,
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = safe_PDF(wo, wi, data, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = static_cast<const T *>(this)->eval(wo, wi, data, mode);
                return {f, wi, PDF_val, BxDFFlags::Reflection};
            }

            ND_XPU_INLINE bool match_flags(BxDFFlags bxdf_flags) {
                static constexpr auto flags = T::flags();
                return ((flags & bxdf_flags) == flags) && valid;
            }
        };
    }
}