//
// Created by Zero on 2021/2/4.
//


#pragma once

#include <cstdint>
#include <type_traits>
#include "../header.h"

namespace luminous {

    inline namespace scalar {

        using uchar = uint8_t;
        using ushort = uint16_t;
        using uint = uint32_t;

        template<typename T>
        struct IsScalar : std::false_type {};

#define MAKE_IS_SCALAR_TRUE(Type) template<> struct IsScalar<Type> : std::true_type {};

        MAKE_IS_SCALAR_TRUE(bool)
        MAKE_IS_SCALAR_TRUE(float)
        MAKE_IS_SCALAR_TRUE(double)
        MAKE_IS_SCALAR_TRUE(int8_t)
        MAKE_IS_SCALAR_TRUE(uint8_t)
        MAKE_IS_SCALAR_TRUE(int16_t)
        MAKE_IS_SCALAR_TRUE(uint16_t)
        MAKE_IS_SCALAR_TRUE(int32_t)
        MAKE_IS_SCALAR_TRUE(uint32_t)
#undef MAKE_IS_SCALAR_TRUE

        template<typename T>
        constexpr auto is_scalar = IsScalar<T>::value;

    }

    inline namespace functor {
        [[nodiscard]] XPU constexpr auto next_pow_of_two(uint v) noexcept {
            v--;
            v |= v >> 1u;
            v |= v >> 2u;
            v |= v >> 4u;
            v |= v >> 8u;
            v |= v >> 16u;
            v++;
            return v;
        }

        [[nodiscard]] XPU constexpr float radians(float deg) noexcept {
            return deg * constant::Pi / 180.0f;
        }
        [[nodiscard]] XPU constexpr float degrees(float rad) noexcept {
            return rad * constant::invPi * 180.0f;
        }

        template<typename T, typename F>
        [[nodiscard]] XPU constexpr auto select(bool pred, T t, F f) noexcept {
            return pred ? t : f;
        }

        template<typename A, typename B>
        [[nodiscard]] XPU constexpr auto lerp(float t, A a, B b) noexcept {
            return a + t * (b - a);
        }

        template <typename T, typename U, typename V>
        [[nodiscard]] XPU constexpr T clamp(T val, U low, V high) noexcept {
            if (val < low)
                return low;
            else if (val > high)
                return high;
            else
                return val;
        }

#if defined(_GNU_SOURCE)
        inline void sincos(float theta, float *sin, float *cos) {
            ::sincosf(theta, sin, cos);
        }
#else
        inline XPU void sincos(float theta, float *_sin, float *_cos) {
            *_sin = sinf(theta);
            *_cos = cosf(theta);
        }
#endif

        inline XPU float safe_sqrt(float x) noexcept {
            return sqrt(std::max(x, 0.f));
        }

        inline XPU float safe_acos(float x) noexcept {
            return acos(clamp(x, -1.f, 1.f));
        }

        inline XPU float safe_asin(float x) noexcept {
            return asin(clamp(x, -1.f, 1.f));
        }

        template <typename T>
        inline XPU constexpr auto sqr(T v) {
            return v * v;
        }

        template <int n>
        XPU constexpr float Pow(float v) {
            if constexpr (n < 0) {
                return 1 / Pow<-n>(v);
            } else if constexpr (n == 1) {
                return v;
            } else if constexpr (n == 0) {
                return 1;
            }
            float n2 = Pow<n / 2>(v);
            return n2 * n2 * Pow<n & 1>(v);
        }

        template <typename IntegerType>
        XPU_INLINE IntegerType round_up(IntegerType x, IntegerType y) {
            return ( ( x + y - 1 ) / y ) * y;
        }

        XPU_INLINE bool is_power_of_two(uint32_t i) noexcept { return (i & (i-1)) == 0; }

        XPU_INLINE bool is_power_of_two(int32_t i) noexcept { return i > 0 && (i & (i-1)) == 0; }

        XPU_INLINE bool is_power_of_two(uint64_t i) noexcept { return (i & (i-1)) == 0; }

        XPU_INLINE bool is_power_of_two(int64_t i) noexcept { return i > 0 && (i & (i-1)) == 0; }

    }
}