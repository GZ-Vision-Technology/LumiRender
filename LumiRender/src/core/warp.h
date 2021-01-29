//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "math/math_util.h"
#include "owl/common.h"

namespace luminous::render {
    inline namespace warp {

        using vec3f = owl::vec3fa;
        using owl::vec2f;

        inline __both__ vec2f square_to_disk(const vec2f u) {
            auto r = sqrt(u.x);
            auto theta = constant::_2Pi * u.y;
            return vec2f(r * cos(theta), r * sin(theta));
        }

        inline __both__ float uniform_disk_pdf() {
            return constant::invPi;
        }

        inline __both__ vec3f square_to_cosine_hemisphere(const vec2f u) {
            auto d = square_to_disk(u);
            auto z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
            return vec3f(d.x, d.y, z);
        }

        inline __both__ float cosine_hemisphere_pdf(float cos_theta) {
            return cos_theta * constant::invPi;
        }

        inline __both__ vec3f square_to_cone(const vec2f u, float cos_theta_max) {
            float cos_theta = (1 - u.x) + u.x * cos_theta_max;
            float sin_theta = sqrt(1 - cos_theta * cos_theta);
            float phi = constant::_2Pi * u.y;
            return vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        }

        inline __both__ float uniform_cone_pdf(float cos_theta_max) {
            return 1 / (constant::_2Pi * (1 - cos_theta_max));
        }

        inline __both__ vec2f square_to_triangle(const vec2f u) {
            auto su0 = sqrt(u.x);
            return vec2f(1 - su0, u.x * su0);
        }

        inline __both__ vec3f square_to_sphere(vec2f u) {
            float z = 1 - 2 * u[0];
            float r = std::sqrt(std::max((float)0, (float)1 - z * z));
            float phi = 2 * Pi * u[1];
            return vec3f(r * std::cos(phi), r * std::sin(phi), z);
        }

        inline __both__ float uniform_sphere_pdf() {
            return constant::inv4Pi;
        }

        inline __both__ vec3f square_to_hemisphere(const vec2f sample) {
            float z = sample.x;
            float tmp = math::safe_sqrt(1.0f - z*z);

            float sinPhi, cosPhi;
            math::sincos(2.0f * constant::Pi * sample.y, &sinPhi, &cosPhi);

            return vec3f(cosPhi * tmp, sinPhi * tmp, z);
        }

        inline __both__ float uniform_hemisphere_pdf() {
            return constant::inv2Pi;
        }

        inline __both__ float balance_heuristic(int nf,
                                         float f_pdf,
                                         int ng,
                                         float g_pdf) {
            return (nf * f_pdf) / (nf * f_pdf + ng * g_pdf);
        }

        inline __both__ float power_heuristic(int nf,
                                     float f_pdf,
                                     int ng,
                                     float g_pdf) {
            float f = nf * f_pdf, g = ng * g_pdf;
            return (f * f) / (f * f + g * g);
        }

        inline __both__ float mis_weight(int nf,
                                float f_pdf,
                                int ng,
                                float g_pdf) {
            return balance_heuristic(nf, f_pdf, ng, g_pdf);
        }

        inline __both__ float mis_weight(float f_pdf, float g_pdf) {
            return mis_weight(1, f_pdf, 1, g_pdf);
        }
    }
}