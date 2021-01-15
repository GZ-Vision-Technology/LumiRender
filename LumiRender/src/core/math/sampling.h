//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "math_util.h"

namespace luminous::render {
    inline namespace sampling {

        /**
         * 均匀分布可得p(x,y) = 1/π
         * p(x,y) = p(θ,r)/r 8式
         * 又由8式，可得
         * p(θ,r) = r/π
         * 由边缘概率密度函数公式可得
         * p(r) = ∫[0,2π]p(θ,r)dθ = 2r
         * p(θ|r) = p(θ,r)/p(r) = 1/2π
         * θ与r相互独立 p(θ|r) = 1/2π = p(θ)
         * 对p(θ)积分可得
         * P(θ) = θ/2π
         * 对p(r)积分可得
         * P(r) = r^2
         *
         * a,b为[0,1]的均匀分布随机数
         * r = √(a)
         * θ = 2πb
         */
        inline float2 uniform_sample_disk(const float2 u) {
            auto r = sqrt(u.x);
            auto theta = constant::_2Pi * u.y;
            return make_float2(r * cos(theta), r * sin(theta));
        }

        inline float3 cosine_sample_hemisphere(const float2 u) {
            auto d = uniform_sample_disk(u);
            auto z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
            return make_float3(d.x, d.y, z);
        }

        /**
         * p(θ, φ) = sinθ p(w)
         * p(θ, φ) = pθ(θ) * pφ(φ)
         * pφ(φ) = 1/2π
         *
         * p(w) = pθ(θ)/(2π * sinθ) 为常数
         * 所以p(θ)/sinθ为常数
         * 假设p(θ) = c * sinθ
         *
         * 1 = ∫[0,θmax]p(θ)dθ
         * 求得 c = 1/(1 - cosθmax)
         * p(θ) = sinθ/(1 - cosθmax)
         * p(w) = p(θ, φ)/sinθ = pθ(θ) * pφ(φ)/sinθ = 1/(2π(1-cosθmax))
         */
        inline float uniform_cone_pdf(float cos_theta_max) {
            return 1 / (constant::_2Pi * (1 - cos_theta_max));
        }

        /**
         *  均匀采样三角形
         * 转换为均匀采样直角三角形，直角边分别为uv，长度为1
         * 三角形面积为s = 1/2
         * p(u, v) = 2
         * p(u) = ∫[0, 1-u]p(u, v)dv = 2(1 - u)
         * p(u|v) = p(u,v)/p(u) = 1/(1 - u)
         * 积分得
         * P(u) = ∫[0, u]p(u')du' = 2u - u^2
         * P(v) = ∫[0, v]p(u|v')dv' = v/(1 - u)
         *
         * ab为均匀分布的随机数
         * 对P(u) P(v)求反函数，得
         * u = 1 - √a
         * v = b * √a
         * @param  u 均匀二维随机变量
         * @return   三角形内部uv坐标
         */
        inline float2 uniform_sample_triangle(const float2 u) {
            auto su0 = sqrt(u.x);
            return make_float2(1 - su0, u.x * su0);
        }

        /**
         * 均匀采样一个圆锥，默认圆锥的中心轴为(0,0,1)，圆锥顶点为坐标原点
         * 可以认为圆锥采样是sphere，hemisphere采样的一般化
         * 当圆锥采样的θmax为π/2时，圆锥采样为hemisphere采样
         * 当圆锥采样的θmax为π时，圆锥采样为sphere采样
         *
         * p(θ) = sinθ/(1 - cosθmax)
         *
         * 积分计算得到累积分布函数
         * P(θ) = (cosθ - 1)/(cosθmax - 1)
         * P(φ) = φ/2π
         *
         * a,b为[0,1]的均匀分布随机数
         * cosθ = (1 - a) + a * cosθmax
         * φ = 2πb
         *
         * sinθ = sqrt(1 - cosθ * cosθ)
         *
         * x = sinθcosφ
         * y = sinθsinφ
         * z = cosθ
         */
        inline float3 uniform_sample_cone(const float2 u, float cos_theta_max) {
            float cos_theta = (1 - u.x) + u.x * cos_theta_max;
            float sin_theta = sqrt(1 - cos_theta * cos_theta);
            float phi = constant::_2Pi * u.y;
            return make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        }

        inline float cosine_hemisphere_pdf(float cos_theta) {
            return cos_theta * constant::invPi;
        }

        inline float3 uniform_sample_sphere(float2 u) {
            float z = 1 - 2 * u[0];
            float r = std::sqrt(std::max((float)0, (float)1 - z * z));
            float phi = 2 * Pi * u[1];
            return make_float3(r * std::cos(phi), r * std::sin(phi), z);
        }

        inline float uniform_sphere_pdf() {
            return constant::inv4Pi;
        }

        inline float uniform_hemisphere_pdf() {
            return constant::inv2Pi;
        }

        inline float balance_heuristic(int nf,
                                         float f_pdf,
                                         int ng,
                                         float g_pdf) {
            return (nf * f_pdf) / (nf * f_pdf + ng * g_pdf);
        }

        inline float power_heuristic(int nf,
                                     float f_pdf,
                                     int ng,
                                     float g_pdf) {
            float f = nf * f_pdf, g = ng * g_pdf;
            return (f * f) / (f * f + g * g);
        }

        inline float mis_weight(int nf,
                                float f_pdf,
                                int ng,
                                float g_pdf) {
            return balance_heuristic(nf, f_pdf, ng, g_pdf);
        }

        inline float mis_weight(float f_pdf, float g_pdf) {
            return mis_weight(1, f_pdf, 1, g_pdf);
        }
    }
}