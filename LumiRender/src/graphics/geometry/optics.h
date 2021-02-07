//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    inline namespace optics {
        template<typename T>
        XPU inline Vector<T, 3> reflect(const Vector<T, 3> wo, const Vector<T, 3> n) {
            return -wo + 2 * dot(wo, n) * n;
        }

        template<typename T>
        XPU inline bool refract(Vector<T, 3> wi, Vector<T, 3> n, T eta, Vector<T, 3> *wt) {
            T cosTheta_i = dot(n, wi);
            T sin2Theta_i = max<T>(0, 1 - cosTheta_i * cosTheta_i);
            T sin2Theta_t = sin2Theta_i / sqr(eta);
            if (sin2Theta_t >= 1) {
                return false;
            }
            T cosTheta_t = safe_sqrt(1 - sin2Theta_t);
            *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector<T, 3>(n);
            return true;
        }
    }
}