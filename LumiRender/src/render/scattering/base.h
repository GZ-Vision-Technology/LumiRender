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

            LM_XPU void disable() {
                PDF = -1.f;
            }

            LM_ND_XPU bool valid() const {
                return PDF > 0.f;
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
        protected:
            BxDFFlags _flags{Unset};
        public:
            LM_XPU explicit BxDF(BxDFFlags flags = Unset): _flags(flags) {}

            ND_XPU_INLINE float weight(BSDFHelper helper, float Fr) const {
                return 1.f;
            }

            LM_ND_XPU Spectrum safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ?
                       static_cast<const T *>(this)->eval(wo, wi, helper) :
                       Spectrum{0.f};
            }

            ND_XPU_INLINE float safe_PDF(float3 wo, float3 wi, BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? cosine_hemisphere_PDF(Frame::abs_cos_theta(wi)) : 0.f;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                          TransportMode mode = TransportMode::Radiance) const {
                float3 wi = square_to_cosine_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = safe_PDF(wo, wi, helper, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = static_cast<const T *>(this)->eval(wo, wi, helper, mode);
                return {f, wi, PDF_val, BxDFFlags::DiffRefl};
            }

            ND_XPU_INLINE BxDFFlags flags() const {
                return _flags;
            }

            ND_XPU_INLINE bool match_flags(BxDFFlags bxdf_flags) {
                return ((_flags & bxdf_flags) == _flags);
            }
        };

        template<typename T>
        struct ColoredBxDF : public BxDF<T> {
        protected:
            float _r{0.f};
            float _g{0.f};
            float _b{0.f};
        public:
            using BxDF<T>::BxDF;

            LM_XPU explicit ColoredBxDF(Spectrum color, BxDFFlags flags)
                    : _r(color.R()), _g(color.G()), _b(color.B()),
                      BxDF<T>(color.is_black() ? Unset : flags) {}

            ND_XPU_INLINE float weight(BSDFHelper helper, float Fr) const {
                return spectrum().luminance();
            }

            ND_XPU_INLINE Spectrum spectrum() const {
                return Spectrum{_r, _g, _b};
            }

            ND_XPU_INLINE Spectrum color(BSDFHelper helper) const {
                return spectrum();
            }
        };
    }
}