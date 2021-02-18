//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "vector_types.h"

namespace luminous {
    inline namespace math {
        /**
         * 跟复数一样，四元数用来表示旋转
         * q = w + xi + yj + zk
         *其中 i^2 = j^2 = k^2 = ijk = -1
         *实部为w，虚部为x,y,z
         单位四元数为 x^2 + y^2 + z^2 + w^2 = 1

         *四元数的乘法法则与复数相似
         *qq' = (qw + qxi + qyj + qxk) * (q'w + q'xi + q'yj + q'xk)
         *展开整理之后
         *(qq')xyz = cross(qxyz, q'xyz) + qw * q'xyz + q'w * qxyz
         *(qq')w = qw * q'w - dot(qxyz, q'xyz)

         * 四元数用法
         *一个点p在绕某个向量单位v旋转2θ之后p',其中旋转四元数为q = (cosθ, v * sinθ)，q为单位四元数
         * 则满足p' = q * p * p^-1
         */
        struct Quaternion {
            float3 v;
            float w;
            XPU Quaternion(float3 v = make_float3(0), float w = 1) :
            v(v),
            w(w) {

            }

            XPU Quaternion conj() {
                return Quaternion(-v, w);
            }

            XPU Quaternion &operator += (const Quaternion &q) {
                v += q.v;
                w += q.w;
                return *this;
            }

            XPU friend Quaternion operator + (const Quaternion &q1, const Quaternion &q2) {
                Quaternion ret = q1;
                return ret += q2;
            }

            Quaternion &operator -= (const Quaternion &q) {
                v -= q.v;
                w -= q.w;
                return *this;
            }

            XPU Quaternion operator - () const {
                Quaternion ret;
                ret.v = -v;
                ret.w = -w;
                return ret;
            }

            XPU friend Quaternion operator - (const Quaternion &q1, const Quaternion &q2) {
                Quaternion ret = q1;
                return ret -= q2;
            }

            XPU Quaternion &operator *= (float f) {
                v *= f;
                w *= f;
                return *this;
            }

            XPU Quaternion operator * (float f) const {
                Quaternion ret = *this;
                ret.v *= f;
                ret.w *= f;
                return ret;
            }

            XPU Quaternion &operator/=(float f) {
                v /= f;
                w /= f;
                return *this;
            }

            XPU Quaternion operator / (float f) const {
                Quaternion ret = *this;
                ret.v /= f;
                ret.w /= f;
                return ret;
            }

            XPU bool has_nan() const {
                return v.has_nan() || is_nan(w);
            }

            [[nodiscard]] std::string to_string() const {
                auto theta = degrees(acos(w));
                return string_printf("quaternion:{ v:%s, angle:%f}", v.to_string().c_str(), (theta * 2));
            }
        };

        [[nodiscard]] XPU_INLINE Quaternion operator*(float f, const Quaternion &q) {
            return q * f;
        }
    }
}