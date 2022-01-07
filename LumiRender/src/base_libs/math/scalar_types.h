//
// Created by Zero on 2021/2/4.
//


#pragma once

#include <cstdint>
#include <type_traits>
#include "constants.h"
#include "../header.h"

namespace luminous {

    inline namespace scalar {

        using uchar = uint8_t;
        using ushort = uint16_t;
        using uint = uint32_t;

        using index_t = uint32_t;

        template<typename T>
        LM_ND_XPU bool is_valid_index(T index) {
            return index != T(-1);
        }

        template<typename T>
        struct IsScalar : std::false_type {
        };

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
        LM_NODISCARD LM_XPU constexpr auto next_pow_of_two(uint v) noexcept {
            v--;
            v |= v >> 1u;
            v |= v >> 2u;
            v |= v >> 4u;
            v |= v >> 8u;
            v |= v >> 16u;
            v++;
            return v;
        }

        LM_NODISCARD LM_XPU constexpr float radians(float deg) noexcept {
            return deg * constant::Pi / 180.0f;
        }

        LM_NODISCARD LM_XPU constexpr float degrees(float rad) noexcept {
            return rad * constant::invPi * 180.0f;
        }

        template<typename T, typename F>
        LM_NODISCARD LM_XPU constexpr auto select(bool pred, T t, F f) noexcept {
            return pred ? t : f;
        }

        template<typename A, typename B>
        LM_NODISCARD LM_XPU constexpr auto lerp(float t, A a, B b) noexcept {
            return a + t * (b - a);
        }

        template<typename T, typename U, typename V>
        LM_NODISCARD LM_XPU constexpr T clamp(T val, U low, V high) noexcept {
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

        inline LM_XPU void sincos(float theta, float *_sin, float *_cos) {
            *_sin = sinf(theta);
            *_cos = cosf(theta);
        }

#endif

        template<typename FloatType>
        ND_XPU_INLINE FloatType safe_sqrt(FloatType x) noexcept {
            if constexpr (std::is_same_v<FloatType, float>) {
                return std::sqrtf(std::max(x, 0.f));
            } else {
                return std::sqrt(std::max(x, 0.f));
            }
        }

        template<typename T>
        LM_ND_XPU T sign(T val) {
            return val >= 0 ? T(1) : T(-1);
        }

        template<typename T>
        LM_ND_XPU T abs(T val) {
            if constexpr(std::is_same_v<T, float>) {
                return ::fabsf(val);
            }
            return ::fabs(val);
        }

        template<typename FloatType>
        ND_XPU_INLINE FloatType safe_acos(FloatType x) noexcept {
            if constexpr (std::is_same_v<FloatType, float>) {
                return std::acosf(clamp(x, -1.f, 1.f));
            } else {
                return std::acos(clamp(x, -1.f, 1.f));
            }
        }

        template<typename FloatType>
        ND_XPU_INLINE FloatType safe_asin(FloatType x) noexcept {
            if constexpr (std::is_same_v<FloatType, float>) {
                return std::asinf(clamp(x, -1.f, 1.f));
            } else {
                return std::asin(clamp(x, -1.f, 1.f));
            }
        }

        template<typename T>
        inline LM_XPU constexpr auto sqr(T v) {
            return v * v;
        }

        template<int n>
        LM_XPU constexpr float Pow(float v) {
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

        template<uint32_t N>
        ND_XPU_INLINE unsigned int tea(uint32_t val0, uint32_t val1) {
            uint32_t v0 = val0;
            uint32_t v1 = val1;
            uint32_t s0 = 0;

            for (uint32_t n = 0; n < N; n++) {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }

            return v0;
        }

        ND_XPU_INLINE int log2_int(uint32_t v) {
#if defined(_MSC_VER)
            unsigned long lz = 0;
            if (_BitScanReverse(&lz, v)) {
                return lz;
            }
            return 0;
#else
            return 31 - __builtin_clz(v);
#endif
        }

        ND_XPU_INLINE int log2_int(int32_t v) {
            return log2_int((uint32_t) v);
        }

        ND_XPU_INLINE int log2_int(uint64_t v) {
#if defined(_MSC_VER)
            unsigned long lz = 0;
#if defined(_WIN64)
            _BitScanReverse64(&lz, v);
#else
            if  (_BitScanReverse(&lz, v >> 32))
                lz += 32;
            else
                _BitScanReverse(&lz, v & 0xffffffff);
#endif
            return lz;
#else
            return 63 - __builtin_clzll(v);
#endif
        }

        ND_XPU_INLINE int log2_int(int64_t v) {
            return log2_int((uint64_t) v);
        }

        template<typename IntegerType>
        LM_XPU_INLINE IntegerType round_up(IntegerType x, IntegerType y) {
            return ((x + y - 1) / y) * y;
        }

        LM_XPU_INLINE int32_t round_up_POT(int32_t v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return v + 1;
        }

        template<typename T>
        LM_XPU_INLINE T Mod(T a, T b) {
            T result = a - (a / b) * b;
            return (T) ((result < 0) ? result + b : result);
        }

        template<>
        LM_XPU_INLINE float Mod(float a, float b) {
            return std::fmodf(a, b);
        }

        template<>
        LM_XPU_INLINE double Mod(double a, double b) {
            return std::fmod(a, b);
        }

        LM_XPU_INLINE int64_t round_up_POT(int64_t v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            return v + 1;
        }

        LM_XPU_INLINE bool is_power_of_two(uint32_t i) noexcept { return (i & (i - 1)) == 0; }

        LM_XPU_INLINE bool is_power_of_two(int32_t i) noexcept { return i > 0 && (i & (i - 1)) == 0; }

        LM_XPU_INLINE bool is_power_of_two(uint64_t i) noexcept { return (i & (i - 1)) == 0; }

        LM_XPU_INLINE bool is_power_of_two(int64_t i) noexcept { return i > 0 && (i & (i - 1)) == 0; }

    }
}