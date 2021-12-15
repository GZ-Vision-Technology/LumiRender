//
// Created by Zero on 29/11/2021.
//

#include "dielectric.h"

namespace luminous {
    inline namespace render {

        Spectrum DielectricBxDF::eval(float3 wo, float3 wi, TransportMode mode) const {
            if (_microfacet.effectively_smooth()) {
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

            if (dot(wh, wi) * cos_theta_i < 0
                || dot(wh, wo) * cos_theta_o < 0) {
                return 0.f;
            }

            if (reflect) {
                eta_p = cos_theta_o > 0 ? _eta : (1.f / _eta);
                float F = fresnel_dielectric(dot(wo, wh), eta_p);
                if (F == 0.f) {
                    return 0.f;
                }
                float fr = _microfacet.BRDF(wo, wh, wi, F, cos_theta_i, cos_theta_o, mode);
                return fr * Kr;
            } else {
                eta_p = cos_theta_o > 0 ? _eta : (1.f / _eta);
                float F = fresnel_dielectric(dot(wo, wh), eta_p);
                if (F == 1) {
                    return 0.f;
                }
                float ft = _microfacet.BTDF(wo, wh, wi, 1 - F, eta_p, cos_theta_o, cos_theta_i, mode);
                return ft * Kt;
            }
            return {};
        }

        float DielectricBxDF::PDF(float3 wo, float3 wi, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            if (_microfacet.effectively_smooth()) {
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

            if (dot(wh, wi) * cos_theta_i < 0
                || dot(wh, wo) * cos_theta_o < 0) {
                return 0.f;
            }

            float R = fresnel_dielectric(dot(wo, wh), eta_p);
            float T = 1 - R;

            float pr = R, pt = T;
            if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
                pr = 0.f;
            }
            if (!(sample_flags & BxDFReflTransFlags::Transmission)) {
                pt = 0.f;
            }
            if (pr == 0 && pt == 0) {
                return 0.f;
            }

            float PDF = 0.f;
            if (reflect) {
                PDF = _microfacet.PDF_wi_reflection(wo, wh);
                PDF = PDF * pr / (pt + pr);
            } else {
                PDF = _microfacet.PDF_wi_transmission(wo, wh, wi, eta_p);
                PDF = PDF * pt / (pt + pr);
            }
            return PDF;
        }

        BSDFSample DielectricBxDF::sample_f(float3 wo, float uc, float2 u, TransportMode mode,
                                                            BxDFReflTransFlags sample_flags) const {
            if (_microfacet.effectively_smooth()) {
                float R = fresnel_dielectric(Frame::cos_theta(wo), _eta);
                float T = 1 - R;
                float pr = R, pt = T;
                if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
                    pr = 0.f;
                }
                if (!(sample_flags & BxDFReflTransFlags::Transmission)) {
                    pt = 0.f;
                }
                if (pr == 0 && pt == 0) {
                    return {};
                }
                if (uc < pr / (pr + pt)) {
                    // reflection
                    float3 wi = make_float3(-wo.x, -wo.y, wo.z);
                    float fr = R / Frame::abs_cos_theta(wi);
                    Spectrum val = fr * Kr;
                    return {val, wi, pr / (pr + pt), Reflection, _eta};
                } else {
                    // transmission
                    float3 wi{};
                    float eta_p = 0;
                    bool valid = refract(wo, make_float3(0, 0, 1), _eta, &eta_p, &wi);
                    if (!valid) {
                        return {};
                    }

                    float ft = T / Frame::abs_cos_theta(wi);
                    if (mode == TransportMode::Radiance) {
                        ft = ft / sqr(eta_p);
                    }
                    Spectrum val = ft * Kt;
                    return {val, wi, pt / (pr + pt), Transmission, eta_p};
                }
            } else {
                float3 wh = _microfacet.sample_wh(wo, u);
                float R = fresnel_dielectric(dot(wo, wh), _eta);
                float T = 1 - R;
                float pr = R, pt = T;
                if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
                    pr = 0.f;
                }
                if (!(sample_flags & BxDFReflTransFlags::Transmission)) {
                    pt = 0.f;
                }
                if (pr == 0 && pt == 0) {
                    return {};
                }
                float PDF{};
                if (uc < pr / (pr + pt)) {
                    // reflection
                    float3 wi = reflect(wo, wh);
                    float cos_theta_i = Frame::cos_theta(wi);
                    float cos_theta_o = Frame::cos_theta(wo);
                    if (!same_hemisphere(wi, wo)) {
                        return {};
                    }
                    PDF = _microfacet.PDF_wi_reflection(wo, wh) * pr / (pr + pt);
                    float fr = _microfacet.BRDF(wo, wh, wi, R, cos_theta_i, cos_theta_o, mode);
                    Spectrum val = fr * Kr;
                    return {val, wi, PDF, Reflection, _eta};
                } else {
                    // transmission
                    float eta_p{};
                    float3 wi{};
                    bool valid = refract(wo, wh, _eta, &eta_p, &wi);
                    if (!valid) {
                        return {};
                    }
                    if (same_hemisphere(wo, wi)) {
                        return {};
                    }
                    float cos_theta_o = Frame::cos_theta(wo);
                    float cos_theta_i = Frame::cos_theta(wi);
                    PDF = _microfacet.PDF_wi_transmission(wo, wh, wi, eta_p) * pt / (pr + pt);
                    float ft = _microfacet.BTDF(wo, wh, wi, 1 - R, cos_theta_o, cos_theta_i, eta_p, mode);
                    Spectrum val = ft * Kt;
                    if (dot(wo, wh) < 0) {
                        return {};
                    }
                    return {val, wi, PDF, Transmission, _eta};
                }
            }
            return {};
        }

        BxDFFlags DielectricBxDF::flags() const {
            if (_eta == 1)
                return BxDFFlags::Transmission;
            else
                return BxDFFlags::Reflection | BxDFFlags::Transmission |
                       (_microfacet.effectively_smooth() ? BxDFFlags::Specular : BxDFFlags::Glossy);
        }
    }
}