//
// Created by Zero on 03/01/2022.
//

#include "diffuse_scatter.h"

namespace luminous {
    inline namespace render {

        Spectrum OrenNayar::eval(float3 wo, float3 wi, BSDFHelper helper, TransportMode mode) const {
            float sin_theta_i = Frame::sin_theta(wi);
            float sin_theta_o = Frame::sin_theta(wo);

            float max_cos = 0;

            if (sin_theta_i > 1e-4 && sin_theta_o > 1e-4) {
                float sinPhiI = Frame::sin_phi(wi), cosPhiI = Frame::cos_phi(wi);
                float sinPhiO = Frame::sin_phi(wo), cosPhiO = Frame::cos_phi(wo);
                float d_cos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
                max_cos = std::max(0.f, d_cos);
            }

            float sin_alpha, tan_beta;
            bool condition = Frame::abs_cos_theta(wi) > Frame::abs_cos_theta(wo);
            sin_alpha = condition ? sin_theta_o : sin_theta_i;
            tan_beta = condition ?
                    sin_theta_i / Frame::abs_cos_theta(wi) :
                    sin_theta_o / Frame::abs_cos_theta(wo);
            float2 AB = helper.AB();
            float factor = (AB.x + AB.y * max_cos * sin_alpha * tan_beta);
            return color(helper) * invPi * factor;
        }
    }
}