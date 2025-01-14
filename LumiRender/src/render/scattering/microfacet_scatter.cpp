//
// Created by Zero on 03/01/2022.
//

#include "microfacet_scatter.h"
#include "../textures/image_texture.h"

namespace luminous {
    inline namespace render {

        BSDFSample MicrofacetReflection::_sample_f_color(float3 wo, float uc, float2 u, Spectrum Fr, BSDFHelper helper,
                                                         Spectrum color, TransportMode mode) const {
            float3 wh = _microfacet.sample_wh(wo, u);
            if (dot(wh, wo) < 0) {
                return {};
            }
            float3 wi = reflect(wo, wh);
            if (!same_hemisphere(wi, wo)) {
                return {};
            }
            float PDF = _microfacet.PDF_wi_reflection(wo, wh);
            Spectrum val = _f(wo, wi, helper, color, mode);
            return {val, wi, PDF, flags(), helper.eta()};
        }

        float MicrofacetReflection::weight(BSDFHelper helper, Spectrum Fr) const {
            return luminance(ColoredBxDF::weight(helper, Fr) * Fr);
        }

        Spectrum MicrofacetReflection::_f(float3 wo, float3 wi, BSDFHelper helper,
                                          Spectrum color, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            float cos_theta_i = Frame::cos_theta(wi);
            float3 wh = normalize(wo + wi);
            wh = face_forward(wh, make_float3(0, 0, 1));
            Spectrum F = helper.eval_fresnel(abs_dot(wo, wh));
            Spectrum fr = _microfacet.BRDF(wo, wh, wi, F, cos_theta_i, cos_theta_o, mode);
            return fr * color;
        }

        Spectrum MicrofacetReflection::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            return _f(wo, wi, helper, color(helper), mode);
        }

        Spectrum MicrofacetReflection::safe_eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            if (!same_hemisphere(wi, wo)) {
                return {0.f};
            }
            helper.correct_eta(cos_theta_o);
            return eval(wo, wi, helper, mode);
        }

        float MicrofacetReflection::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float3 wh = normalize(wo + wi);
            return _microfacet.PDF_wi_reflection(wo, wh);
        }

        float MicrofacetReflection::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            if (!same_hemisphere(wo, wi)) {
                return 0.f;
            }
            return PDF(wo, wi, helper, mode);
        }

        BSDFSample MicrofacetReflection::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr,
                                                   BSDFHelper helper, TransportMode mode) const {
            return _sample_f_color(wo, uc, u, Fr, helper, color(helper), mode);
        }

        BSDFSample MicrofacetReflection::sample_f(float3 wo, float uc, float2 u,
                                                  BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            helper.correct_eta(cos_theta_o);
            return _sample_f(wo, uc, u, 0.f, helper, mode);
        }

        // MicrofacetTransmission
        Spectrum MicrofacetTransmission::_f(float3 wo, float3 wi, BSDFHelper helper,
                                            Spectrum color, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            float cos_theta_i = Frame::cos_theta(wi);
            using eta_type = decltype(helper.eta());
            float3 wh = normalize(wo + wi * helper.eta());
            if (dot(wo, wh) * dot(wi, wh) > 0) {
                return {0.f};
            }
            wh = face_forward(wh, make_float3(0, 0, 1));
            Spectrum F = helper.eval_fresnel(abs_dot(wo, wh));
            Spectrum tr = _microfacet.BTDF(wo, wh, wi, Spectrum(1.f) - F, cos_theta_i,
                                           cos_theta_o, helper.eta(), mode);
            return tr * color;
        }

        BSDFSample MicrofacetTransmission::_sample_f_color(float3 wo, float uc, float2 u, Spectrum Fr,
                                                           BSDFHelper helper, Spectrum color,
                                                           TransportMode mode) const {
            float3 wh = _microfacet.sample_wh(wo, u);
            if (dot(wh, wo) < 0) {
                return {};
            }
            float3 wi{};
            bool valid = refract(wo, wh, helper.eta(), &wi);
            if (!valid || same_hemisphere(wo, wi)) {
                return {};
            }
            float PDF = _microfacet.PDF_wi_transmission(wo, wh, wi, helper.eta());
            Spectrum val = _f(wo, wi, helper, color, mode);
            return {val, wi, PDF, flags(), helper.eta()};
        }

        Spectrum MicrofacetTransmission::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            return _f(wo, wi, helper, color(helper), mode);
        }

        Spectrum MicrofacetTransmission::safe_eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            if (same_hemisphere(wi, wo)) {
                return {0.f};
            }
            helper.correct_eta(cos_theta_o);
            return eval(wo, wi, helper, mode);
        }

        float MicrofacetTransmission::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            float cos_theta_i = Frame::cos_theta(wi);
            float3 wh = normalize(wo + wi * helper.eta());
            if (dot(wo, wh) * dot(wi, wh) > 0) {
                return 0.f;
            }
            wh = face_forward(wh, make_float3(0, 0, 1));
            return _microfacet.PDF_wi_transmission(wo, wh, wi, helper.eta());
        }

        float MicrofacetTransmission::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            if (same_hemisphere(wo, wi)) {
                return 0.f;
            }
            helper.correct_eta(cos_theta_o);
            return PDF(wo, wi, helper, mode);
        }

        BSDFSample MicrofacetTransmission::_sample_f(float3 wo, float uc, float2 u,
                                                     Spectrum Fr, BSDFHelper helper,
                                                     TransportMode mode) const {
            return _sample_f_color(wo, uc, u, Fr, helper, color(helper), mode);
        }

        BSDFSample MicrofacetTransmission::sample_f(float3 wo, float uc, float2 u,
                                                    BSDFHelper helper, TransportMode mode) const {
            float cos_theta_o = Frame::cos_theta(wo);
            helper.correct_eta(cos_theta_o);
            return _sample_f(wo, uc, u, 0.f, helper, mode);
        }

        // MicrofacetFresnel
        float MicrofacetFresnel::PDF_specular(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float3 wh = normalize(wo + wi);
            float pdf_wh = _microfacet.PDF_wh(wo, wh);
            return pdf_wh / (4 * dot(wo, wh));
        }

        Spectrum MicrofacetFresnel::eval_specular(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float3 wh = wi + wo;
            if (is_zero(wh)) {
                return {0.f};
            }
            wh = normalize(wh);
            Spectrum specular = _microfacet.D(wh) /
                    (4 * abs_dot(wi, wh) * std::max(Frame::abs_cos_theta(wi), Frame::abs_cos_theta(wo))) *
                    schlick_fresnel(dot(wi, wh), helper);
            return specular;
        }

        float MicrofacetFresnel::PDF_diffuse(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            return cosine_hemisphere_PDF(Frame::abs_cos_theta(wi));
        }

        Spectrum MicrofacetFresnel::eval_diffuse(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            Spectrum Rd = color(helper);
            Spectrum Rs = _spec;
            Spectrum diffuse = (28.f / (23.f * Pi)) * Rd * (Spectrum(1.f) - Rs) *
                    (1 - Pow<5>(1 - .5f * Frame::abs_cos_theta(wi))) *
                    (1 - Pow<5>(1 - .5f * Frame::abs_cos_theta(wo)));
            return diffuse;
        }

        Spectrum MicrofacetFresnel::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            if (!same_hemisphere(wo, wi)) {
                return {0.f};
            }
            return safe_eval(wo, wi, helper, mode);
        }

        Spectrum MicrofacetFresnel::safe_eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            Spectrum diffuse = eval_diffuse(wo, wi, helper, mode);
            Spectrum specular = eval_specular(wo, wi, helper, mode);

            return diffuse + specular;
        }

        Spectrum MicrofacetFresnel::schlick_fresnel(float cos_theta, BSDFHelper helper) const {
            return fresnel_schlick(_spec, cos_theta);
        }

        float MicrofacetFresnel::PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            if (!same_hemisphere(wo, wi)) {
                return 0.f;
            }
            return safe_PDF(wo, wi, helper, mode);
        }

        float MicrofacetFresnel::safe_PDF(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            return 0.5 * (PDF_diffuse(wo, wi, helper, mode) + PDF_specular(wo, wi, helper, mode));
        }

        BSDFSample MicrofacetFresnel::sample_f(float3 wo, float uc, float2 u,
                                               BSDFHelper helper, TransportMode mode) const {
            float3 wi{0.f};
            Spectrum f_val{0.f};
            float pdf = 0.f;
            float fr = helper.eval_fresnel(Frame::abs_cos_theta(wo))[0];
            if (uc > fr) {
                wi = square_to_cosine_hemisphere(u);
                wi.z = select(wo.z < 0, -wi.z, wi.z);
                pdf = PDF_diffuse(wo, wi, helper, mode) * (1 - fr);
                f_val = eval_diffuse(wo, wi, helper, mode);
            } else {
                float3 wh = _microfacet.sample_wh(wo, u);
                wi = reflect(wo, wh);
                if (!same_hemisphere(wi, wo)) {
                    return BSDFSample(f_val, wi, 0, flags(), helper.eta());
                }
                pdf = PDF_specular(wo, wi, helper, mode) * fr;
                f_val = eval_specular(wo, wi, helper, mode);
            }
            return BSDFSample(f_val, wi, pdf, flags(), helper.eta());
        }

        Spectrum ClothMicrofacetFresnel::eval_diffuse(float3 wo, float3 wi, BSDFHelper data, TransportMode mode) const {

            float alpha = data.clearcoat_alpha();

            float2 spec_wo_scaling = _spec_albedo->eval(float2(Frame::cos_theta(wo), alpha));
            float2 spec_wi_scaling = _spec_albedo->eval(float2(Frame::cos_theta(wi), alpha));
            float2 spec_avg_scaling = _spec_albedo_avg->eval(float2(alpha, 1));
            Spectrum R0 = data.R0();

            return color(data) * (Spectrum(1.0f) - (R0 * spec_wo_scaling.x + spec_wo_scaling.y)) * (Spectrum(1.0f) - (R0 * spec_wi_scaling.x + spec_wi_scaling.y)) /
                   (Pi * (Spectrum(1.0f) - (R0 * spec_avg_scaling.x + spec_avg_scaling.y)));
        }

        Spectrum ClothMicrofacetFresnel::eval_specluar(float3 wo, float3 wi, BSDFHelper data, TransportMode mode) const {
            float3 wh = normalize(wo + wi);
            Spectrum Fr = fresnel_schlick(data.R0(), std::max(dot(wh, wi), .0f));
            return _microfacet.BRDF(wo, wh, wi, Fr, Frame::cos_theta(wi), Frame::cos_theta(wo), mode);
        }


        Spectrum ClothMicrofacetFresnel::eval(float3 wo, float3 wi, BSDFHelper helper,
                                              TransportMode mode) const {

            if (!same_hemisphere(wo, wi)) {
                return {0.f};
            }
            return safe_eval(wo, wi, helper, mode);
        }

        Spectrum ClothMicrofacetFresnel::safe_eval(float3 wo, float3 wi, BSDFHelper helper,
                                                   TransportMode mode) const {
            return eval_diffuse(wo, wi, helper, mode) + eval_specluar(wo, wi, helper, mode);
        }

        float ClothMicrofacetFresnel::PDF(float3 wo, float3 wi,
                                          BSDFHelper helper,
                                          TransportMode mode) const {
            if (!same_hemisphere(wo, wi)) {
                return 0.f;
            }
            return safe_PDF(wo, wi, helper, mode);
        }

        float ClothMicrofacetFresnel::safe_PDF(float3 wo, float3 wi,
                                               BSDFHelper helper,
                                               TransportMode mode) const {
            // Do uniform sampling
            return uniform_hemisphere_PDF();
        }

        BSDFSample ClothMicrofacetFresnel::sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                                    TransportMode mode) const {
            // Do uniform sampling
            float3 wi{0.f};
            Spectrum f_val{0.f};
            float pdf = 0.f;

            float3 wh = _microfacet.sample_wh(wo, u);
            wi = reflect(wo, wh);
            if (!same_hemisphere(wo, wi))
                return BSDFSample(f_val, wi, 0, flags(), helper.eta());

            f_val = safe_eval(wo, wo, helper, mode);
            pdf = _microfacet.PDF_wh(wo, wh) / std::max(4.0f * dot(wh, wi), 1.0e-3f);
            return BSDFSample(f_val, wi, 0, flags(), helper.eta());
        }
    }
}