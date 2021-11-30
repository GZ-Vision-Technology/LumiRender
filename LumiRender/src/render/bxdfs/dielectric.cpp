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
            float cos_theta_o = Frame::cos_theta(wo);
            float cos_theta_i = Frame::cos_theta(wi);
            bool reflect = cos_theta_i * cos_theta_o > 0;
            float eta_p = 1;
            if (!reflect) {
                eta_p = cos_theta_o > 0 ? _eta : (1.f / _eta);
            }
            float3 wh = normalize(wo + wi * eta_p);
            if (cos_theta_i == 0 || cos_theta_o == 0 || length_squared(wh) == 0) {
                return 0.f;
            }

            wh = face_forward(wh, make_float3(0, 0, 1));

            if (dot(wh, wi) * cos_theta_i < 0 || dot(wh, wo) * cos_theta_o < 0) {
                return 0.f;
            }

            if (reflect) {
                eta_p = cos_theta_o > 0 ? _eta : (1.f / _eta);
                float F = fresnel_dielectric(dot(wo, wh), eta_p);
                float ret = _distribution.D(wh) * F * _distribution.G(wo, wi)
                            / std::abs(4 * cos_theta_o * cos_theta_i);
                return ret * Kr;
            } else {
                eta_p = cos_theta_i < 0 ? _eta : (1.f / _eta);
                float F = fresnel_dielectric(dot(wo, wh), eta_p);
                float denom = sqr(dot(wi, wh) * eta_p + dot(wo, wh)) * cos_theta_i * cos_theta_o;

                float numerator = _distribution.D(wh) * (1 - F) * _distribution.G(wo, wi) *
                                  std::abs(dot(wi, wh) * dot(wo, wh));
                float ft = numerator / denom;


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