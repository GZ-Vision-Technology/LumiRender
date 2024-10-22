//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace math {

        LM_XPU_CONSTANT_VISIBILITY constexpr float FloatOneMinusEpsilon = 0.99999994;
        LM_XPU_CONSTANT_VISIBILITY constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;

        template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        ND_XPU_INLINE bool is_nan(T v) {
#if defined(__CUDACC__)
            return ::isnan(v);
#else
            return std::isnan(v);
#endif
        }

        template<typename T, std::enable_if_t<std::is_integral<T>::value, bool> = false>
        ND_XPU_INLINE bool is_nan(T v) {
            return false;
        }

        template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        ND_XPU_INLINE bool is_inf(T v) {
#if defined(__CUDACC__)
            return ::isinf(v);
#else
            return std::isinf(v);
#endif
        }

        template<typename T, std::enable_if_t<std::is_integral<T>::value, bool> = false>
        ND_XPU_INLINE bool is_inf(T v) {
            return false;
        }

        template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        ND_XPU_INLINE bool is_nature_number(T v) {
            return !is_inf(v) && !is_nan(v);
        }

        template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        ND_XPU_INLINE bool is_invalid(T v) {
            return is_inf(v) || is_nan(v);
        }

        LM_XPU_INLINE float FMA(float a, float b, float c) {
            return std::fmaf(a, b, c);
        }

        LM_XPU_INLINE double FMA(double a, double b, double c) {
            return std::fma(a, b, c);
        }

    } // luminous::math
} // luminous