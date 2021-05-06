//
// Created by Zero on 2021/1/29.
//


#pragma once


#include "base.h"


namespace luminous {
    inline namespace render {

        class IdealDiffuse {
        private:
            float4 _R = make_float4(-1.f);

            NDSC_XPU float4 _eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                return _R * constant::invPi;
            }

            NDSC_XPU bool valid() const {
                return any(_R != make_float4(-1.f));
            }

        public:
            XPU IdealDiffuse() = default;

            XPU IdealDiffuse(float4 R) : _R(R) {}

            NDSC_XPU float4 base_color() const {
                return _R;
            }

            NDSC_XPU float4 eval(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance) const {
                return same_hemisphere(wo, wi) ? _eval(wo, wi) : make_float4(0.f);
            }

            NDSC_XPU float PDF(float3 wo, float3 wi, TransportMode mode = TransportMode::Radiance,
                               BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                return cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
            }

            NDSC_XPU lstd::optional<BSDFSample> sample_f(float3 wo, float uc, float2 u,
                                                         TransportMode mode = TransportMode::Radiance,
                                                         BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const {
                if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
                    return {};
                }
                float3 wi = square_to_cosine_hemisphere(u);
                if (wo.z < 0) {
                    wi.z *= -1;
                }
                float PDF = cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
                float4 f = _eval(wo, wi, mode);
                return BSDFSample(f, wi, PDF, BxDFFlags::Reflection);
            }

            NDSC_XPU BxDFFlags flags() const {
                return valid() ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset;
            }

            NDSC std::string to_string() const {
                LUMINOUS_TO_STRING("IdealDiffuse R : %s", _R.to_string().c_str());
            }
        };
    } // luminous::render
} //luminous