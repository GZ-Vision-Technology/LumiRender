//
// Created by Zero on 29/11/2021.
//

#include "microfacet.h"
#include "base_libs/geometry/frame.h"
#include "base_libs/geometry/util.h"

namespace luminous {
    inline namespace render {

        float MicrofacetDistribution::D(const float3 &wh) const {
            // When ¦È is close to 90, tan¦È is infinity
            float tan_theta_2 = Frame::tan_theta_2(wh);
            if (is_inf(tan_theta_2)) {
                return 0.f;
            }
            float cos_theta_4 = sqr(Frame::cos_theta_2(wh));
            if (cos_theta_4 < 1e-16f) {
                return 0.f;
            }
            switch (_type) {
                case GGX: {
                    float e = tan_theta_2 * (sqr(Frame::cos_phi(wh) / _alpha_x) + sqr(Frame::sin_phi(wh) / _alpha_y));
                    return 1.f / (Pi * _alpha_x * _alpha_y * cos_theta_4 * sqr(1 + e));
                }
                case Beckmann: {
                    return std::exp(-tan_theta_2 * (Frame::cos_phi_2(wh) / sqr(_alpha_x) +
                                                    Frame::sin_phi_2(wh) / sqr(_alpha_y))) /
                           (Pi * _alpha_x * _alpha_y * cos_theta_4);
                }
                default:
                    break;
            }
            LM_ASSERT(0, "unknown type %d", int(_type));
            return 0;
        }

        float MicrofacetDistribution::lambda(const float3 &w) const {
            switch (_type) {
                case GGX: {
                    float abs_tan_theta = std::abs(Frame::tan_theta(w));
                    if (std::isinf(abs_tan_theta)) {
                        return 0.f;
                    }
                    float alpha = std::sqrt(Frame::cos_phi_2(w) * sqr(_alpha_x) +
                                            Frame::sin_phi_2(w) * sqr(_alpha_y));
                    return (-1 + std::sqrt(1.f + sqr(alpha * abs_tan_theta))) / 2;
                }
                case Beckmann: {
                    float abs_tan_theta = std::abs(Frame::tan_theta(w));
                    if (std::isinf(abs_tan_theta)) {
                        return 0.f;
                    }
                    float alpha = std::sqrt(Frame::cos_phi_2(w) * sqr(_alpha_x) +
                                            Frame::sin_phi_2(w) * sqr(_alpha_y));
                    float a = 1.f / (alpha * abs_tan_theta);
                    if (a >= 1.6f) {
                        return 0.f;
                    }
                    return (1 - 1.259f * a + 0.396f * sqr(a)) / (3.535f * a + 2.181f * sqr(a));
                }
                default:
                    break;
            }
            LM_ASSERT(0, "unknown type %d", int(_type));
            return 0;
        }

        float3 MicrofacetDistribution::sample_wh(const float3 &wo, const float2 &u) const {
            switch (_type) {
                case GGX: {
                    float cos_theta = 0, phi = _2Pi * u[1];
                    if (_alpha_x == _alpha_y) {
                        float tanTheta2 = _alpha_x * _alpha_x * u[0] / (1.0f - u[0]);
                        cos_theta = 1 / std::sqrt(1 + tanTheta2);
                    } else {
                        phi = std::atan(_alpha_y / _alpha_x * std::tan(_2Pi * u[1] + PiOver2));
                        if (u[1] > .5f) {
                            phi += Pi;
                        }
                        float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                        const float alpha2 = 1.f / (sqr(cos_phi / _alpha_x) + sqr(sin_phi / _alpha_y));
                        float tan_theta_2 = alpha2 * u[0] / (1 - u[0]);
                        cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                    }
                    float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                    float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                    if (!same_hemisphere(wo, wh)) {
                        wh = -wh;
                    }
                    return wh;
                }
                case Beckmann: {
                    float tan_theta_2, phi;
                    if (_alpha_x == _alpha_y) {
                        float log_sample = std::log(1 - u[0]);
                        DCHECK(!std::isinf(log_sample));
                        tan_theta_2 = -_alpha_x * _alpha_x * log_sample;
                        phi = u[1] * _2Pi;
                    } else {
                        float log_sample = std::log(1 - u[0]);
                        DCHECK(!std::isinf(log_sample));
                        phi = std::atan(_alpha_y / _alpha_x *
                                std::tan(_2Pi * u[1] + PiOver2));
                        if (u[1] > 0.5f) {
                            phi += Pi;
                        }
                        float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
                        tan_theta_2 = -log_sample / (sqr(cos_phi / _alpha_x) + sqr(sin_phi / _alpha_y));
                    }

                    float cos_theta = 1 / std::sqrt(1 + tan_theta_2);
                    float sin_theta = std::sqrt(std::max(0.f, 1 - sqr(cos_theta)));
                    float3 wh = spherical_direction(sin_theta, cos_theta, phi);
                    if (!same_hemisphere(wo, wh)) {
                        wh = -wh;
                    }
                    return wh;
                }
                default:
                    break;
            }
            LM_ASSERT(0, "unknown type %d", int(_type));
            return {};
        }

        float MicrofacetDistribution::PDF_dir(const float3 &wo, const float3 &wh) const {
            return D(wh) * Frame::abs_cos_theta(wh);
        }
    }
}