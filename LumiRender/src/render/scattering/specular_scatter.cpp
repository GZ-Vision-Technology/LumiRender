//
// Created by Zero on 03/01/2022.
//

#include "specular_scatter.h"

namespace luminous {
    inline namespace render {

        BSDFSample SpecularReflection::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr,
                                                 BSDFHelper data, TransportMode mode) const {
            float3 wi = make_float3(-wo.x, -wo.y, wo.z);
            Spectrum val = Fr * Spectrum(data.color()) / Frame::abs_cos_theta(wi);
            float PDF = 1.f;
            return {val, wi, PDF, BxDFFlags::SpecRefl, data.eta()};
        }

        BSDFSample SpecularReflection::sample_f(float3 wo, float uc, float2 u,
                                                BSDFHelper data, TransportMode mode) const {

            float3 wi = make_float3(-wo.x, -wo.y, wo.z);
            data.correct_eta(Frame::cos_theta(wo));
            auto Fr = data.eval_fresnel(Frame::abs_cos_theta(wo));
            return _sample_f(wo, uc, u, Fr, data, mode);
        }

        BSDFSample SpecularTransmission::_sample_f(float3 wo, float uc, float2 u, Spectrum Fr, BSDFHelper data,
                                                   TransportMode mode) const {
            float3 wi{};
            float3 n = make_float3(0, 0, 1);
            bool valid = refract(wo, face_forward(n, wo), data.eta(), &wi);
            if (!valid) {
                return {};
            }
            Spectrum ft = (Spectrum(1.f) - Fr) / Frame::abs_cos_theta(wi);
            float factor = cal_factor(mode, data.eta());
            Spectrum val = ft * Spectrum(data.color()) * factor;
            return {val, wi, 1, SpecTrans, data.eta()};
        }

        BSDFSample SpecularTransmission::sample_f(float3 wo, float uc, float2 u,
                                                  BSDFHelper data, TransportMode mode) const {
            float3 wi{};
            data.correct_eta(Frame::cos_theta(wo));
            float3 n = make_float3(0, 0, 1);
            bool valid = refract(wo, face_forward(n, wo), data.eta(), &wi);
            if (!valid) {
                return {};
            }
            auto Fr = data.eval_fresnel(Frame::abs_cos_theta(wo))[0];
            return _sample_f(wo, uc, u, Fr, data, mode);
        }
    }
}