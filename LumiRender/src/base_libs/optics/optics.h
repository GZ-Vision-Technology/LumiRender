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
        ND_XPU_INLINE Vector<T, 3> reflect(const Vector<T, 3> wo, const Vector<T, 3> n) {
            return -wo + 2 * dot(wo, n) * n;
        }

        template<typename T>
        ND_XPU_INLINE bool refract(Vector<T, 3> wi, Vector<T, 3> n, T eta, T *eta_p,
                                   Vector<T, 3> *wt) {
            CHECK_UNIT_VEC(wi)
            T cos_theta_i = dot(n, wi);
            // Potentially flip interface orientation for Snell's law
            if (cos_theta_i < 0) {
                eta = 1 / eta;
                cos_theta_i = -cos_theta_i;
                n = -n;
            }

            // Compute $\cos\,\theta_\roman{t}$ using Snell's law
            T sin_theta_i_2 = std::max<T>(0, 1 - sqr(cos_theta_i));
            T sin_theta_t_2 = sin_theta_i_2 / sqr(eta);
            // Handle total internal reflection case
            if (sin_theta_t_2 >= 1) {
                return false;
            }

            T cos_theta_t = safe_sqrt(1 - sin_theta_t_2);

            *wt = normalize(-wi / eta + (cos_theta_i / eta - cos_theta_t) * n);
            // Provide relative IOR along ray to caller
            if (eta_p) {
                *eta_p = eta;
            }

            return true;
        }

        template<typename T>
        LM_XPU inline bool refract(Vector<T, 3> wi, Vector<T, 3> n, T eta, Vector<T, 3> *wt) {
            T cos_theta_i = dot(n, wi);
            DCHECK(cos_theta_i > 0);
            T sin_theta_i_2 = max<T>(0, 1 - sqr(cos_theta_i));
            T sin_theta_t_2 = sin_theta_i_2 / sqr(eta);
            if (sin_theta_t_2 >= 1) {
                return false;
            }
            T cos_theta_t = safe_sqrt(1 - sin_theta_t_2);
            *wt = -wi / eta + (cos_theta_i / eta - cos_theta_t) * n;
            return true;
        }

        template<typename T>
        LM_ND_XPU T correct_eta(float cos_theta_o, T eta) {
            return cos_theta_o > 0 ? eta : rcp(eta);
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

        ND_XPU_INLINE float fresnel_dielectric(float abs_cos_theta_i, float eta) {
            abs_cos_theta_i = clamp(abs_cos_theta_i, -1, 1);

            float sin_theta_i_2 = 1 - sqr(abs_cos_theta_i);
            float sin_theta_t_2 = sin_theta_i_2 / sqr(eta);
            if (sin_theta_t_2 >= 1) {
                return 1.f;
            }
            float cos_theta_t = safe_sqrt(1 - sin_theta_t_2);

            float r_parl = (eta * abs_cos_theta_i - cos_theta_t) / (eta * abs_cos_theta_i + cos_theta_t);
            float r_perp = (abs_cos_theta_i - eta * cos_theta_t) / (abs_cos_theta_i + eta * cos_theta_t);
            return (sqr(r_parl) + sqr(r_perp)) / 2;
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