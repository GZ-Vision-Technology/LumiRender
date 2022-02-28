//
// Created by Zero on 27/12/2021.
//

#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {
        namespace disney {

            LM_ND_XPU Spectrum diffuse_f(Spectrum color, float3 wo, float3 wi, TransportMode mode) {
                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                return color * invPi * (1 - 0.5f * Fo) * (1 - 0.5f * Fi);
            }

            LM_ND_XPU Spectrum retro_f(Spectrum color, float roughness,
                                       float3 wo, float3 wi, TransportMode mode) {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Rr = 2 * roughness * sqr(cos_theta_d);

                return color * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
            }

            LM_ND_XPU Spectrum sheen_f(Spectrum color, float3 wo, float3 wi, TransportMode mode) {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);
                return color * schlick_weight(cos_theta_d);
            }

            LM_ND_XPU Spectrum fake_ss_f(Spectrum color, float roughness,
                                         float3 wo, float3 wi, TransportMode mode) {
                float3 wh = wi + wo;
                //todo optimize branch
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fss90 = sqr(cos_theta_d) * roughness;
                float Fo = schlick_weight(Frame::abs_cos_theta(wo)),
                        Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Fss = lerp(Fo, 1.f, Fss90) * lerp(Fi, 1.f, Fss90);
                float ss = 1.25f * (Fss * (1 / (Frame::abs_cos_theta(wo) + Frame::abs_cos_theta(wi)) - .5f) + .5f);

                return color * invPi * ss;
            }

            ND_XPU_INLINE float GTR1(float cos_theta, float alpha) {
                float alpha2 = sqr(alpha);
                return (alpha2 - 1) /
                       (Pi * std::log(alpha2) * (1 + (alpha2 - 1) * sqr(cos_theta)));
            }

            ND_XPU_INLINE float smithG_GGX(float cos_theta, float alpha) {
                float alpha2 = sqr(alpha);
                float cos_theta_2 = sqr(cos_theta);
                return 1 / (cos_theta + sqrt(alpha2 + cos_theta_2 - alpha2 * cos_theta_2));
            }

            Spectrum Diffuse::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return diffuse_f(spectrum(), wo, wi, mode);
            }

            Spectrum FakeSS::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return fake_ss_f(spectrum(), helper.roughness(), wo, wi, mode);
            }

            float Retro::weight(BSDFHelper helper, Spectrum Fr) const {
                return luminance(spectrum() * Fr);
            }

            float Retro::safe_PDF(float3 wo, float3 wi, BSDFHelper helper,
                                  TransportMode mode) const {
                return same_hemisphere(wo, wi) ? uniform_hemisphere_PDF() : 0.f;
            }

            BSDFSample Retro::sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                       TransportMode mode) const {
                float3 wi = square_to_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = safe_PDF(wo, wi, helper, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = this->eval(wo, wi, helper, mode);
                return {f, wi, PDF_val, BxDFFlags::DiffRefl};
            }

            Spectrum Retro::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return retro_f(spectrum(), helper.roughness(), wo, wi, mode);
            }

            // sheen
            Spectrum Sheen::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return sheen_f(spectrum(), wo, wi, mode);
            }

            float Sheen::weight(BSDFHelper helper, Spectrum Fr) const {
                return luminance(spectrum() * Fr);
            }

            float Sheen::safe_PDF(float3 wo, float3 wi, BSDFHelper helper,
                                  TransportMode mode) const {
                return same_hemisphere(wo, wi) ? uniform_hemisphere_PDF() : 0.f;
            }

            BSDFSample Sheen::sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                       TransportMode mode) const {
                float3 wi = square_to_hemisphere(u);
                wi.z = wo.z < 0 ? -wi.z : wi.z;
                float PDF_val = safe_PDF(wo, wi, helper, mode);
                if (PDF_val == 0.f) {
                    return {};
                }
                Spectrum f = this->eval(wo, wi, helper, mode);
                return {f, wi, PDF_val, BxDFFlags::DiffRefl};
            }

            Spectrum DiffuseLobes::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return diffuse_f(spectrum(), wo, wi, mode) +
                       retro_f(_retro, helper.roughness(), wo, wi, mode) +
                       sheen_f(_sheen, wo, wi, mode) +
                       fake_ss_f(_fake_ss, helper.roughness(), wo, wi, mode);
            }

            // Clearcoat
            Spectrum Clearcoat::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.clearcoat_alpha());
                float Fr = fresnel_schlick(0.04f, dot(wo, wh));
                float Gr = smithG_GGX(Frame::abs_cos_theta(wo), 0.25f)
                           * smithG_GGX(Frame::abs_cos_theta(wi), 0.25f);
                Spectrum ret = _weight * Gr * Fr * Dr / 4;
                DCHECK(!has_invalid(ret))
                return ret;
            }

            float Clearcoat::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return 0.f;
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.clearcoat_alpha());
                return Dr * Frame::abs_cos_theta(wh) / (4 * dot(wo, wh));
            }

            float Clearcoat::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                if (!same_hemisphere(wo, wi)) {
                    return 0;
                }
                return PDF(wo, wi, helper, mode);
            }

            BSDFSample Clearcoat::sample_f(float3 wo, float uc, float2 u,
                                           BSDFHelper helper, TransportMode mode) const {
                if (wo.z == 0) {
                    return {};
                }
                float alpha2 = sqr(helper.clearcoat_alpha());

                float cos_theta = safe_sqrt((1 - std::pow(alpha2, 1 - u[0])) / (1 - alpha2));
                float sin_theta = safe_sqrt(1 - sqr(cos_theta));
                float phi = 2 * Pi * u[1];
                float3 wh = spherical_direction(sin_theta, cos_theta, phi);

                wh = same_hemisphere(wo, wh) ? wh : -wh;
                float3 wi = reflect(wo, wh);
                if (!same_hemisphere(wo, wi)) {
                    return {};
                }
                float pdf = PDF(wo, wi, helper, mode);
                Spectrum f_val = eval(wo, wi, helper, mode);
                return {f_val, wi, pdf, BxDFFlags::GlossyRefl};
            }

        }
    }
}