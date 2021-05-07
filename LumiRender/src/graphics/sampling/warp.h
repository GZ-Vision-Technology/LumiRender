//
// Created by Zero on 2021/2/5.
//


#pragma once

#include "../math/common.h"

namespace luminous {
    inline namespace sampling {

        NDSC_XPU_INLINE float2 square_to_disk(const float2 u) {
            auto r = sqrt(u.x);
            auto theta = constant::_2Pi * u.y;
            return make_float2(r * cos(theta), r * sin(theta));
        }

        NDSC_XPU_INLINE float uniform_disk_PDF() {
            return constant::invPi;
        }

        NDSC_XPU_INLINE float3 square_to_cosine_hemisphere(const float2 u) {
            auto d = square_to_disk(u);
            auto z = sqrt(std::max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
            return make_float3(d.x, d.y, z);
        }

        NDSC_XPU_INLINE float cosine_hemisphere_PDF(float cos_theta) {
            return cos_theta * constant::invPi;
        }

        NDSC_XPU_INLINE float3 square_to_cone(const float2 u, float cos_theta_max) {
            float cos_theta = (1 - u.x) + u.x * cos_theta_max;
            float sin_theta = sqrt(1 - cos_theta * cos_theta);
            float phi = constant::_2Pi * u.y;
            return make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        }

        NDSC_XPU_INLINE float uniform_cone_PDF(float cos_theta_max) {
            return 1 / (constant::_2Pi * (1 - cos_theta_max));
        }

        NDSC_XPU_INLINE float2 square_to_triangle(const float2 u) {
            auto su0 = sqrt(u.x);
            return make_float2(1 - su0, u.x * su0);
        }

        NDSC_XPU_INLINE float3 square_to_sphere(float2 u) {
            float z = 1 - 2 * u[0];
            float r = std::sqrt(std::max((float)0, (float)1 - z * z));
            float phi = 2 * Pi * u[1];
            return make_float3(r * std::cos(phi), r * std::sin(phi), z);
        }

        NDSC_XPU_INLINE float uniform_sphere_PDF() {
            return constant::inv4Pi;
        }

        /**
         * p(dir) = p(pos) * r^2 / cos��
         * @return
         */
        NDSC_XPU_INLINE float PDF_dir(float PDF_pos, float3 normal, float3 wo) {
            float cos_theta = abs_dot(normal, normalize(wo));
            return PDF_pos * length_squared(wo) / cos_theta;
        }

        NDSC_XPU_INLINE float3 square_to_hemisphere(const float2 sample) {
            float z = sample.x;
            float tmp = safe_sqrt(1.0f - z*z);

            float sinPhi, cosPhi;
            sincos(2.0f * constant::Pi * sample.y, &sinPhi, &cosPhi);

            return make_float3(cosPhi * tmp, sinPhi * tmp, z);
        }

        NDSC_XPU_INLINE float uniform_hemisphere_PDF() {
            return constant::inv2Pi;
        }

        NDSC_XPU_INLINE float balance_heuristic(int nf,
                                           float f_PDF,
                                           int ng,
                                           float g_PDF) {
            return (nf * f_PDF) / (nf * f_PDF + ng * g_PDF);
        }

        NDSC_XPU_INLINE float power_heuristic(int nf,
                                         float f_PDF,
                                         int ng,
                                         float g_PDF) {
            float f = nf * f_PDF, g = ng * g_PDF;
            return (f * f) / (f * f + g * g);
        }

        NDSC_XPU_INLINE float MIS_weight(int nf,
                                    float f_PDF,
                                    int ng,
                                    float g_PDF) {
            return balance_heuristic(nf, f_PDF, ng, g_PDF);
        }

        NDSC_XPU_INLINE float MIS_weight(float f_PDF, float g_PDF) {
            return MIS_weight(1, f_PDF, 1, g_PDF);
        }
    }
}