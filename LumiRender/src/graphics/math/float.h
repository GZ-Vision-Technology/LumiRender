//
// Created by Zero on 2021/2/8.
//


#pragma once

namespace luminous {
    inline namespace math {
        template <typename T>
        inline XPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
        is_nan(T v) {
            return std::isnan(v);
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_integral<T>::value, bool> is_nan(
                T v) {
            return false;
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
        is_inf(T v) {
            return std::isinf(v);
        }

        template <typename T>
        inline XPU typename std::enable_if_t<std::is_integral<T>::value, bool>
                is_inf(T v) {
            return false;
        }

        XPU float FMA(float a, float b, float c) {
            return std::fma(a, b, c);
        }

        XPU double FMA(double a, double b, double c) {
            return std::fma(a, b, c);
        }

    }
}