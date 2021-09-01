//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "vector_types.h"

namespace luminous {
    inline namespace math {
        
        /**
         * as complex number, 
         * quaternion is used to represent rotation
         * q = w + xi + yj + zk
         * among i^2 = j^2 = k^2 = ijk = -1
         * real portion is w，imagine portion x,y,z
         *  x^2 + y^2 + z^2 + w^2 = 1
         * 
         * The multiplication rule the same as complex numbers
         * qq' = (qw + qxi + qyj + qxk) * (q'w + q'xi + q'yj + q'xk)
         * unfold
         * (qq')xyz = cross(qxyz, q'xyz) + qw * q'xyz + q'w * qxyz
         * (qq')w = qw * q'w - dot(qxyz, q'xyz)
         * 
         * A point p is rotated 2 theta around some vector unit v get point p',
         * Where the rotation quaternion is:q = (cosθ, v * sinθ)
         * if q is unit quaterion
         * then p' = q * p * p^-1 is satisfied
         */
        struct Quaternion {
            float3 v;
            float w;
            XPU Quaternion(float3 v = make_float3(0), float w = 1) :
            v(v),
            w(w) {

            }

            NDSC_XPU_INLINE Quaternion conj() {
                return Quaternion(-v, w);
            }

            NDSC_XPU_INLINE Quaternion &operator += (const Quaternion &q) {
                v += q.v;
                w += q.w;
                return *this;
            }

            NDSC_XPU_INLINE friend Quaternion operator + (const Quaternion &q1, const Quaternion &q2) {
                Quaternion ret = q1;
                return ret += q2;
            }

            NDSC_XPU_INLINE Quaternion &operator -= (const Quaternion &q) {
                v -= q.v;
                w -= q.w;
                return *this;
            }

            NDSC_XPU_INLINE Quaternion operator - () const {
                Quaternion ret;
                ret.v = -v;
                ret.w = -w;
                return ret;
            }

            NDSC_XPU_INLINE friend Quaternion operator - (const Quaternion &q1, const Quaternion &q2) {
                Quaternion ret = q1;
                return ret -= q2;
            }

            NDSC_XPU_INLINE Quaternion &operator *= (float f) {
                v *= f;
                w *= f;
                return *this;
            }

            NDSC_XPU_INLINE Quaternion operator * (float f) const {
                Quaternion ret = *this;
                ret.v *= f;
                ret.w *= f;
                return ret;
            }

            NDSC_XPU_INLINE Quaternion &operator/=(float f) {
                v /= f;
                w /= f;
                return *this;
            }

            NDSC_XPU_INLINE Quaternion operator / (float f) const {
                Quaternion ret = *this;
                ret.v /= f;
                ret.w /= f;
                return ret;
            }

            GEN_STRING_FUNC({
                auto theta = degrees(acos(w));
                return string_printf("quaternion:{ v:%s, angle:%f}", v.to_string().c_str(), (theta * 2));
            })
        };

        NDSC_XPU_INLINE bool has_nan(Quaternion q) noexcept {
            return has_nan(q.v) || is_nan(q.w);
        }

        NDSC_XPU_INLINE bool has_inf(Quaternion q) noexcept {
            return has_inf(q.v) || is_inf(q.w);
        }

        NDSC_XPU_INLINE Quaternion operator*(float f, const Quaternion &q) {
            return q * f;
        }
    }
}