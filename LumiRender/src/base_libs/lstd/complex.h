//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/header.h"
#include "base_libs/math/scalar_types.h"
#include <cmath>

namespace luminous {
    inline namespace lstd {

        template<typename T>
        struct Complex {
        public:
            T re, im;
        public:
            LM_XPU explicit Complex(T re) : re(re), im(0) {}

            LM_XPU Complex(T re, T im) : re(re), im(im) {}

            LM_ND_XPU Complex operator-() const { return {-re, -im}; }

            LM_ND_XPU Complex operator+(Complex z) const { return {re + z.re, im + z.im}; }

            LM_ND_XPU Complex operator-(Complex z) const { return {re - z.re, im - z.im}; }

            LM_ND_XPU Complex operator*(Complex z) const {
                return {re * z.re - im * z.im, re * z.im + im * z.re};
            }

            LM_ND_XPU Complex operator/(Complex z) const {
                T scale = 1 / (z.re * z.re + z.im * z.im);
                return {scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im)};
            }

            LM_ND_XPU friend Complex operator+(T value, Complex z) {
                return Complex(value) + z;
            }

            LM_ND_XPU friend Complex operator-(T value, Complex z) {
                return Complex(value) - z;
            }

            LM_ND_XPU friend Complex operator*(T value, Complex z) {
                return Complex(value) * z;
            }

            LM_ND_XPU friend Complex operator*(Complex z, T value) {
                return Complex(value) * z;
            }

            LM_ND_XPU friend Complex operator/(T value, Complex z) {
                return Complex(value) / z;
            }
        };

        ND_XPU_INLINE double copysign(double mag, double sign) {
#ifdef __CUDACC__
            return ::copysign(mag, sign);
#else
            return std::copysign(mag, sign);
#endif
        }

        template<typename T>
        LM_ND_XPU T real(const Complex<T> &z) {
            return z.re;
        }

        template<typename T>
        LM_ND_XPU T imag(const Complex<T> &z) {
            return z.im;
        }

        template<typename T>
        LM_ND_XPU T norm(const Complex<T> &z) {
            return z.re * z.re + z.im * z.im;
        }

        template<typename T>
        LM_ND_XPU T abs(const Complex<T> &z) {
            return std::sqrtf(lstd::norm(z));
        }

        template<typename T>
        LM_ND_XPU Complex<T> sqrt(const Complex<T> &z) {
            T n = lstd::abs(z);
            T t1 = std::sqrtf(T(.5) * (n + functor::abs(z.re)));
            T t2 = T(.5) * z.im / t1;

            if (n == 0)
                return Complex<T>{0};

            if (z.re >= 0)
                return Complex<T>{t1, t2};
            else
                return Complex<T>{functor::abs(t2), (T) lstd::copysign((double) t1, (double) z.im)};
        }
    }
}