//
// Created by Zero on 2020/12/31.
//

#pragma once

#include <core/math/math_util.h>

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
        inline float2 uniform_sample_disk(const float2 &u) {
            auto r = sqrt(u.x);
            auto theta = constant::_2Pi * u.y;
            return make_float2(r * cos(theta), r * sin(theta));
        }

        inline float3 cosine_sample_hemisphere(const float2 &u) {
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
        inline float2 uniform_sample_triangle(const float2 &u) {
            auto su0 = sqrt(u.x);
            return make_float2(1 - su0, u.x * su0);
        }


    }
}