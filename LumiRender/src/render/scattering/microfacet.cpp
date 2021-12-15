//
// Created by Zero on 29/11/2021.
//

#include "microfacet.h"
#include "base_libs/geometry/frame.h"
#include "base_libs/geometry/util.h"

namespace luminous {
    inline namespace render {

//        float Microfacet::D(const float3 &wh) const {
//            // When theta is close to 90, tan theta is infinity
//            float tan_theta_2 = Frame::tan_theta_2(wh);
//            if (is_inf(tan_theta_2)) {
//                return 0.f;
//            }
//            float cos_theta_4 = sqr(Frame::cos_theta_2(wh));
//            if (cos_theta_4 < 1e-16f) {
//                return 0.f;
//            }
//            switch (_type) {
//                case GGX: {
//                    float e = tan_theta_2 * (sqr(Frame::cos_phi(wh) / _alpha_x) + sqr(Frame::sin_phi(wh) / _alpha_y));
//                    float ret = 1.f / (Pi * _alpha_x * _alpha_y * cos_theta_4 * sqr(1 + e));
//                    return ret;
//                }
//                case Beckmann: {
//                    return std::exp(-tan_theta_2 * (Frame::cos_phi_2(wh) / sqr(_alpha_x) +
//                                                    Frame::sin_phi_2(wh) / sqr(_alpha_y))) /
//                           (Pi * _alpha_x * _alpha_y * cos_theta_4);
//                }
//                default:
//                    break;
//            }
//            LM_ASSERT(0, "unknown type %d", int(_type));
//            return 0;
//        }

//        float Microfacet::lambda(const float3 &w) const

//        float3 Microfacet::sample_wh(const float3 &wo, const float2 &u) const

//        float Microfacet::PDF_wh(const float3 &wo, const float3 &wh) const
    }
}