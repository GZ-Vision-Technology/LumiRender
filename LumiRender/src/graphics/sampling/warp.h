//
// Created by Zero on 2021/2/5.
//


#pragma once

namespace luminous {
    inline namespace sampling {

        inline XPU float2 square_to_disk(const float2 u) {
            auto r = sqrt(u.x);
            auto theta = constant::_2Pi * u.y;
            return make_float2(r * cos(theta), r * sin(theta));
        }

        inline XPU float uniform_disk_pdf() {
            return constant::invPi;
        }

        inline XPU float3 square_to_cosine_hemisphere(const float2 u) {
            auto d = square_to_disk(u);
            auto z = sqrt(std::max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
            return make_float3(d.x, d.y, z);
        }

        inline XPU float cosine_hemisphere_PDF(float cos_theta) {
            return cos_theta * constant::invPi;
        }

        inline XPU float3 square_to_cone(const float2 u, float cos_theta_max) {
            float cos_theta = (1 - u.x) + u.x * cos_theta_max;
            float sin_theta = sqrt(1 - cos_theta * cos_theta);
            float phi = constant::_2Pi * u.y;
            return make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        }

        inline XPU float uniform_cone_PDF(float cos_theta_max) {
            return 1 / (constant::_2Pi * (1 - cos_theta_max));
        }

        inline XPU float2 square_to_triangle(const float2 u) {
            auto su0 = sqrt(u.x);
            return make_float2(1 - su0, u.x * su0);
        }

        inline XPU float3 square_to_sphere(float2 u) {
            float z = 1 - 2 * u[0];
            float r = std::sqrt(std::max((float)0, (float)1 - z * z));
            float phi = 2 * Pi * u[1];
            return make_float3(r * std::cos(phi), r * std::sin(phi), z);
        }

        inline XPU float uniform_sphere_PDF() {
            return constant::inv4Pi;
        }

        inline XPU float3 square_to_hemisphere(const float2 sample) {
            float z = sample.x;
            float tmp = safe_sqrt(1.0f - z*z);

            float sinPhi, cosPhi;
            sincos(2.0f * constant::Pi * sample.y, &sinPhi, &cosPhi);

            return make_float3(cosPhi * tmp, sinPhi * tmp, z);
        }

        inline XPU float uniform_hemisphere_PDF() {
            return constant::inv2Pi;
        }

        inline XPU float balance_heuristic(int nf,
                                           float f_PDF,
                                           int ng,
                                           float g_PDF) {
            return (nf * f_PDF) / (nf * f_PDF + ng * g_PDF);
        }

        inline XPU float power_heuristic(int nf,
                                         float f_PDF,
                                         int ng,
                                         float g_PDF) {
            float f = nf * f_PDF, g = ng * g_PDF;
            return (f * f) / (f * f + g * g);
        }

        inline XPU float mis_weight(int nf,
                                    float f_PDF,
                                    int ng,
                                    float g_PDF) {
            return balance_heuristic(nf, f_PDF, ng, g_PDF);
        }

        inline XPU float mis_weight(float f_PDF, float g_PDF) {
            return mis_weight(1, f_PDF, 1, g_PDF);
        }
    }
}