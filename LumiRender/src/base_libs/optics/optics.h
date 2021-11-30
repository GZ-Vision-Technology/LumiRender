//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "../math/common.h"

namespace luminous {
    inline namespace optics {
        template<typename T>
        LM_XPU inline Vector<T, 3> reflect(const Vector<T, 3> wo, const Vector<T, 3> n) {
            return -wo + 2 * dot(wo, n) * n;
        }

        template<typename T>
        LM_XPU inline bool refract(Vector<T, 3> wi, Vector<T, 3> n, T eta, Vector<T, 3> *wt) {
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

        ND_XPU_INLINE float schlick_weight(float cos_theta) {
            float m = clamp(1.f - cos_theta, 0.f, 1.f);
            return Pow<5>(m);
        }

        template<typename T>
        LM_ND_XPU auto fresnel_schlick(T R0, float cos_theta) {
            return lerp(schlick_weight(cos_theta), R0, T{1.f});
        }

        ND_XPU_INLINE float schlick_R0_from_eta(float eta) {
            return sqr(eta - 1) / sqr(eta + 1);
        }

        ND_XPU_INLINE float henyey_greenstein(float cos_theta, float g) {
            float denom = 1 + sqr(g) + 2 * g * cos_theta;
            return inv4Pi * (1 - sqr(g)) / (denom * safe_sqrt(denom));
        }

        ND_XPU_INLINE float fresnel_dielectric(float cos_theta_i, float eta) {
            cos_theta_i = clamp(cos_theta_i, -1, 1);
            if (cos_theta_i < 0) {
                eta = 1 / eta;
                cos_theta_i = -cos_theta_i;
            }

            float sin2Theta_i = 1 - sqr(cos_theta_i);
            float sin2Theta_t = sin2Theta_i / sqr(eta);
            if (sin2Theta_t >= 1)
                return 1.f;
            float cos_theta_t = safe_sqrt(1 - sin2Theta_t);

            float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
            float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
            return (r_parl * r_parl + r_perp * r_perp) / 2;
        }

        ND_XPU_INLINE Spectrum fresnel_conductor(float cos_theta_i, const Spectrum &eta_i,
                                                 const Spectrum &eta_t, const Spectrum &kt) {
            cos_theta_i = clamp(cos_theta_i, -1, 1);
            Spectrum eta = eta_t / eta_i;
            Spectrum eta_k = kt / eta_i;

            float cos_theta_i_2 = sqr(cos_theta_i);
            float sin_theta_i_2 = 1.f - cos_theta_i_2;

            Spectrum t0 = sqr(eta) - sqr(eta_k) - sin_theta_i_2;
            Spectrum a2plusb2 = sqrt(sqr(t0) + 4.f * sqr(eta) * sqr(eta_k));
            Spectrum t1 = a2plusb2 + cos_theta_i_2;
            Spectrum a = sqrt(0.5f * (a2plusb2 + t0));
            Spectrum t2 = 2.f * cos_theta_i * a;
            Spectrum Rs = (t1 - t2) / (t1 + t2);

            Spectrum t3 = cos_theta_i_2 * a2plusb2 + sqr(sin_theta_i_2);
            Spectrum t4 = t2 * sin_theta_i_2;
            Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

            return 0.5f * (Rp + Rs);
        }

    }
}