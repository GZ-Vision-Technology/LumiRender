//
// Created by Zero on 27/12/2021.
//

#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {
        namespace disney {

            Spectrum Diffuse::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float Fo = schlick_weight(Frame::abs_cos_theta(wo));
                float Fi = schlick_weight(Frame::abs_cos_theta(wi));
                return _factor * helper.color() * invPi * (1 - Fo / 2) * (1 - Fi / 2);
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

                return helper.color() * invPi * ss * _factor;
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

                return helper.color() * invPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1)) * _factor;
            }

            Spectrum Sheen::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                float3 wh = wi + wo;
                if (length_squared(wh) == 0.f) {
                    return {0.f};
                }
                wh = normalize(wh);
                float cos_theta_d = dot(wi, wh);

                return helper.color_sheen_tint() * _factor * schlick_weight(cos_theta_d);
            }

            // Clearcoat
            Spectrum Clearcoat::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
                return luminous::Spectrum();
            }

            float Clearcoat::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) {
                return 0;
            }

            float Clearcoat::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) {
                return 0;
            }

            BSDFSample Clearcoat::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr,
                                            BSDFHelper helper, TransportMode mode) {
                return BSDFSample();
            }

            BSDFSample Clearcoat::sample_f(float3 wo, float uc, float2 u,
                                           BSDFHelper helper, TransportMode mode) {
                return BSDFSample();
            }
        }
    }
}