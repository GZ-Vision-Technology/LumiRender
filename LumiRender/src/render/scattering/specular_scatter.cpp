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
    }
}