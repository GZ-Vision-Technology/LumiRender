//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "../math/common.h"
#include "base_libs/lstd/complex.h"
#include "rgb.h"

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
                eta = 1.f / eta;
                cos_theta_i = -cos_theta_i;
            }

            float sin_theta_i_2 = 1 - sqr(cos_theta_i);
            float sin_theta_t_2 = sin_theta_i_2 / sqr(eta);
            if (sin_theta_t_2 >= 1) {
                return 1.f;
            }
            float cos_theta_t = safe_sqrt(1 - sin_theta_t_2);

            float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
            float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
            return (sqr(r_parl) + sqr(r_perp)) / 2;
        }

        ND_XPU_INLINE float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
            return fresnel_dielectric(cos_theta_i, eta_t / eta_i);
        }

        ND_XPU_INLINE float fresnel_complex(float cos_theta_i, lstd::Complex<float> eta) {
            using Complex = lstd::Complex<float>;
            cos_theta_i = clamp(cos_theta_i, 0, 1);
            float sin_theta_i_2 = 1 - sqr(cos_theta_i);
            Complex sin_theta_t_2 = sin_theta_i_2 / sqr(eta);
            Complex cos_theta_t = lstd::sqrt(1.f - sin_theta_t_2);

            Complex r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
            Complex r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
            return (lstd::norm(r_parl) + lstd::norm(r_perp)) * .5f;
        }

        ND_XPU_INLINE float fresnel_complex(float cos_theta_i, float eta, float k) {
            return fresnel_complex(cos_theta_i, lstd::Complex<float>(eta, k));
        }

        ND_XPU_INLINE Spectrum fresnel_complex(float cos_theta_i, Spectrum eta, Spectrum k) {
            Spectrum ret;
            for (int i = 0; i < 3; ++i) {
                ret[i] = fresnel_complex(cos_theta_i, eta[i], k[i]);
            }
            return ret;
        }

        ND_XPU_INLINE Spectrum fresnel_conductor(float cos_theta_i, const Spectrum &eta_i,
                                                 const Spectrum &eta_t, const Spectrum &kt) {
            return fresnel_complex(cos_theta_i, eta_t / eta_i, kt);
        }
    }
}