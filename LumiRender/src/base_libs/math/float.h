//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace math {

#if defined(__CUDACC__)
        #define DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
        #define FloatOneMinusEpsilon float(0x1.fffffep-1)
        #define OneMinusEpsilon FloatOneMinusEpsilon
#else
        static constexpr float FloatOneMinusEpsilon = 0.99999994;
        static constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;
#endif

        template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        inline LM_XPU bool is_nan(T v) {
#if defined(__CUDACC__)
            return ::isnan(v);
#else
            return std::isnan(v);
#endif
        }

        template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = false>
        inline LM_XPU bool is_nan(T v) {
            return false;
        }

        template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = false>
        inline LM_XPU bool is_inf(T v) {
#if defined(__CUDACC__)
            return ::isinf(v);
#else
            return std::isinf(v);
#endif
        }

        template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = false>
        inline LM_XPU bool is_inf(T v) {
            return false;
        }

        LM_XPU_INLINE float FMA(float a, float b, float c) {
            return std::fma(a, b, c);
        }

        LM_XPU_INLINE double FMA(double a, double b, double c) {
            return std::fma(a, b, c);
        }

    } // luminous::math
} // luminous