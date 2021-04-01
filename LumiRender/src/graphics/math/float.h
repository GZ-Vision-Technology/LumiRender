//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace math {
        template <typename T>
        inline XPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
        is_nan(T v) {
#if defined(__CUDACC__)
            return ::isnan(v);
#else
            return std::isnan(v);
#endif
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_integral<T>::value, bool> is_nan(
                T v) {
            return false;
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
        is_inf(T v) {
#if defined(__CUDACC__)
            return ::isinf(v);
#else
            return std::isinf(v);
#endif
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_integral<T>::value, bool>
                is_inf(T v) {
            return false;
        }

        XPU_INLINE float FMA(float a, float b, float c) {
            return std::fma(a, b, c);
        }

        XPU_INLINE double FMA(double a, double b, double c) {
            return std::fma(a, b, c);
        }

    } // luminous::math
} // luminous