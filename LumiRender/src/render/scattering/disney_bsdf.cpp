//
// Created by Zero on 27/12/2021.
//

#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {
        namespace disney {

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
                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                return spectrum() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
            }

            Spectrum FakeSS::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fss90 = sqr(cos_theta_d) * helper.roughness();
                float Fo = schlick_weight(Frame::abs_cos_theta(wo)),
                        Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Fss = lerp(Fo, 1.f, Fss90) * lerp(Fi, 1.f, Fss90);
                float ss = 1.25f * (Fss * (1 / (Frame::abs_cos_theta(wo) + Frame::abs_cos_theta(wi)) - .5f) + .5f);

                return spectrum() * invPi * ss;
            }

            Spectrum Retro::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                float Rr = 2 * helper.roughness() * sqr(cos_theta_d);

                return spectrum() * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
            }

            Spectrum Sheen::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);
                return spectrum() * schlick_weight(cos_theta_d);
            }

            // Clearcoat
            Spectrum Clearcoat::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.gloss());
                float Fr = fresnel_schlick(0.04f, dot(wo, wh));
                float Gr = smithG_GGX(Frame::abs_cos_theta(wo), 0.25f)
                           * smithG_GGX(Frame::abs_cos_theta(wi), 0.25f);
                return _weight * Gr * Fr * Dr / 4;
            }

            float Clearcoat::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return 0.f;
                }
                wh = normalize(wh);

                float Dr = GTR1(Frame::abs_cos_theta(wh), helper.gloss());
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
                float alpha2 = sqr(helper.gloss());

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