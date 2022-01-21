//
// Created by Zero on 03/01/2022.
//

#include "specular_scatter.h"

namespace luminous {
    inline namespace render {

        BSDFSample SpecularReflection::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr,
                                                 BSDFHelper helper, TransportMode mode) const {
            float3 wi = make_float3(-wo.x, -wo.y, wo.z);
            Spectrum val = Fr * spectrum() / Frame::abs_cos_theta(wi);
            float PDF = 1.f;
            return {val, wi, PDF, flags(), helper.eta()};
        }

        BSDFSample SpecularReflection::sample_f(float3 wo, float uc, float2 u,
                                                BSDFHelper helper, TransportMode mode) const {

            float3 wi = make_float3(-wo.x, -wo.y, wo.z);
            helper.correct_eta(Frame::cos_theta(wo));
            auto Fr = helper.eval_fresnel(Frame::abs_cos_theta(wo));
            return _sample_f(wo, uc, u, Fr, helper, mode);
        }

        BSDFSample SpecularTransmission::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr, BSDFHelper helper,
                                                   TransportMode mode) const {
            float3 wi{};
            float3 n = make_float3(0, 0, 1);
            bool valid = refract(wo, face_forward(n, wo), helper.eta(), &wi);
            if (!valid) {
                return {};
            }
            Spectrum ft = (Spectrum(1.f) - Fr) / Frame::abs_cos_theta(wi);
            float factor = cal_factor(mode, helper.eta());
            Spectrum val = ft * spectrum() * factor;
            return {val, wi, 1, flags(), helper.eta()};
        }

        BSDFSample SpecularTransmission::sample_f(float3 wo, float uc, float2 u,
                                                  BSDFHelper helper, TransportMode mode) const {
            float3 wi{};
            helper.correct_eta(Frame::cos_theta(wo));
            float3 n = make_float3(0, 0, 1);
            bool valid = refract(wo, face_forward(n, wo), helper.eta(), &wi);
            if (!valid) {
                return {};
            }
            auto Fr = helper.eval_fresnel(Frame::abs_cos_theta(wo))[0];
            return _sample_f(wo, uc, u, Fr, helper, mode);
        }

        float fresnel_moment1(float eta) {

            float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
                  eta5 = eta4 * eta;
            if (eta < 1)
                return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
                       2.49277f * eta4 - 0.68441f * eta5;
            else
                return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
                       1.27198f * eta4 + 0.12746f * eta5;
        }

        float fresnel_moment2(float eta) {
            float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
                  eta5 = eta4 * eta;
            if (eta < 1) {
                return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
                       0.07883f * eta4 + 0.04860f * eta5;
            } else {
                float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
                return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 +
                       458.843f * r_eta + 404.557f * eta - 189.519f * eta2 +
                       54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
            }
        }

        NormalizedFresnelBxDF::NormalizedFresnelBxDF(float eta) : BxDF(DiffRefl), _eta(eta) {
            _inv_c_mul_pi = 1.0f / ((1 - 2 * fresnel_moment1(rcp(_eta))) * invPi);
        }

        BSDFSample NormalizedFresnelBxDF::sample_f(float3 wo, float uc, float2 u, const BSDFHelper &data, TransportMode mode) const {

            float3 wi = square_to_cosine_hemisphere(u);
            if (wo.z < 0)
                wo.z *= -1;
            return BSDFSample{safe_eval(wo, wi, data, mode), wi, safe_PDF(wo, wi, data, mode), BxDFFlags::DiffRefl};
        }

        Spectrum NormalizedFresnelBxDF::safe_eval(float3 wo, float3 wi, const BSDFHelper &data, TransportMode mode) const {
            if (!same_hemisphere(wo, wi))
                return {};

            Spectrum f = (1 - fresnel_dielectric(Frame::cos_theta(wi), _eta)) * _inv_c_mul_pi;

            if (mode == TransportMode::Radiance)
                f *= sqr(_eta);

            return f;
        }

        float NormalizedFresnelBxDF::safe_PDF(float3 wo, float3 wi, const BSDFHelper &data, TransportMode mode) const {

            return same_hemisphere(wo, wi) ? Frame::abs_cos_theta(wi) * invPi : 0;
        }
    }
}