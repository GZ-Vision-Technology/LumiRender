//
// Created by Zero on 2021/2/5.
//


#pragma once

#include <algorithm>
#include <cmath>
#include "primes.h"
#include "../header.h"

namespace luminous {
    inline namespace constant {

        constexpr auto Pi = 3.14159265358979323846264338327950288f;
        constexpr auto _2Pi = Pi * 2;
        constexpr auto inv2Pi = 1 / _2Pi;
        constexpr auto inv4Pi = 1 / (4 * Pi);
        constexpr auto PiOver2 = 1.57079632679489661923132169163975144f;
        constexpr auto PiOver4 = 0.785398163397448309615660845819875721f;
        constexpr auto invPi = 0.318309886183790671537767526745028724f;
        constexpr auto _2OverPi = 0.636619772367581343075535053490057448f;
        constexpr auto sqrtOf2 = 1.41421356237309504880168872420969808f;
        constexpr auto invSqrtOf2 = 0.707106781186547524400844362104849039f;
        constexpr float float_one_minus_epsilon = 0x1.fffffep-1;
        constexpr float one_minus_epsilon = float_one_minus_epsilon;

        static struct ZeroTy {
            XPU operator double() const { return 0; }

            XPU operator float() const { return 0; }

            XPU operator long long() const { return 0; }

            XPU operator unsigned long long() const { return 0; }

            XPU operator long() const { return 0; }

            XPU operator unsigned long() const { return 0; }

            XPU operator int() const { return 0; }

            XPU operator unsigned int() const { return 0; }

            XPU operator short() const { return 0; }

            XPU operator unsigned short() const { return 0; }

            XPU operator char() const { return 0; }

            XPU operator unsigned char() const { return 0; }
        } zero MAYBE_UNUSED;

        static struct OneTy {
            XPU operator double() const { return 1; }

            XPU operator float() const { return 1; }

            XPU operator long long() const { return 1; }

            XPU operator unsigned long long() const { return 1; }

            XPU operator long() const { return 1; }

            XPU operator unsigned long() const { return 1; }

            XPU operator int() const { return 1; }

            XPU operator unsigned int() const { return 1; }

            XPU operator short() const { return 1; }

            XPU operator unsigned short() const { return 1; }

            XPU operator char() const { return 1; }

            XPU operator unsigned char() const { return 1; }
        } one MAYBE_UNUSED;


        static struct NegInfTy {
#ifdef __CUDA_ARCH__
            __device__ operator          double   ( ) const { return -CUDART_INF; }
            __device__ operator          float    ( ) const { return -CUDART_INF_F; }
#else
            XPU operator double() const { return -std::numeric_limits<double>::infinity(); }

            XPU operator float() const { return -std::numeric_limits<float>::infinity(); }

            XPU operator long long() const { return std::numeric_limits<long long>::min(); }

            XPU operator unsigned long long() const { return std::numeric_limits<unsigned long long>::min(); }

            XPU operator long() const { return std::numeric_limits<long>::min(); }

            XPU operator unsigned long() const { return std::numeric_limits<unsigned long>::min(); }

            XPU operator int() const { return std::numeric_limits<int>::min(); }

            XPU operator unsigned int() const { return std::numeric_limits<unsigned int>::min(); }

            XPU operator short() const { return std::numeric_limits<short>::min(); }

            XPU operator unsigned short() const { return std::numeric_limits<unsigned short>::min(); }

            XPU operator char() const { return std::numeric_limits<char>::min(); }

            XPU operator unsigned char() const { return std::numeric_limits<unsigned char>::min(); }

#endif
        } neg_inf MAYBE_UNUSED;

        static struct PosInfTy {
#ifdef __CUDA_ARCH__
            __device__ operator          double   ( ) const { return CUDART_INF; }
            __device__ operator          float    ( ) const { return CUDART_INF_F; }
#else
            XPU operator double() const { return std::numeric_limits<double>::infinity(); }

            XPU operator float() const { return std::numeric_limits<float>::infinity(); }

            XPU operator long long() const { return std::numeric_limits<long long>::max(); }

            XPU operator unsigned long long() const { return std::numeric_limits<unsigned long long>::max(); }

            XPU operator long() const { return std::numeric_limits<long>::max(); }

            XPU operator unsigned long() const { return std::numeric_limits<unsigned long>::max(); }

            XPU operator int() const { return std::numeric_limits<int>::max(); }

            XPU operator unsigned int() const { return std::numeric_limits<unsigned int>::max(); }

            XPU operator short() const { return std::numeric_limits<short>::max(); }

            XPU operator unsigned short() const { return std::numeric_limits<unsigned short>::max(); }

            XPU operator char() const { return std::numeric_limits<char>::max(); }

            XPU operator unsigned char() const { return std::numeric_limits<unsigned char>::max(); }

#endif
        } inf MAYBE_UNUSED, pos_inf MAYBE_UNUSED;

        static struct NaNTy {
#ifdef __CUDA_ARCH__
            __device__ operator double( ) const { return CUDART_NAN; }
            __device__ operator float ( ) const { return CUDART_NAN_F; }
#else
            XPU operator double() const { return std::numeric_limits<double>::quiet_NaN(); }

            XPU operator float() const { return std::numeric_limits<float>::quiet_NaN(); }

#endif
        } nan MAYBE_UNUSED;

        static struct UlpTy {
#ifdef __CUDA_ARCH__
            // todo
#else
            XPU operator double() const { return std::numeric_limits<double>::epsilon(); }

            XPU operator float() const { return std::numeric_limits<float>::epsilon(); }

#endif
        } ulp MAYBE_UNUSED;


        template<bool is_integer>
        struct limits_traits;

        template<>
        struct limits_traits<true> {
            template<typename T>
            static inline XPU T value_limits_lower(T) { return std::numeric_limits<T>::min(); }

            template<typename T>
            static inline XPU T value_limits_upper(T) { return std::numeric_limits<T>::max(); }
        };

        template<>
        struct limits_traits<false> {
            template<typename T>
            static inline XPU T value_limits_lower(T) { return (T) NegInfTy(); }

            template<typename T>
            static inline XPU T value_limits_upper(T) { return (T) PosInfTy(); }
        };

        /*! lower value of a completely *empty* range [+inf..-inf] */
        template<typename T>
        inline XPU T empty_bounds_lower() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
        }

        /*! upper value of a completely *empty* range [+inf..-inf] */
        template<typename T>
        inline XPU T empty_bounds_upper() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
        }

        /*! lower value of a completely *empty* range [+inf..-inf] */
        template<typename T>
        inline XPU T empty_range_lower() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
        }

        /*! upper value of a completely *empty* range [+inf..-inf] */
        template<typename T>
        inline XPU T empty_range_upper() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
        }

        /*! lower value of a completely open range [-inf..+inf] */
        template<typename T>
        inline XPU T open_range_lower() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
        }

        /*! upper value of a completely open range [-inf..+inf] */
        template<typename T>
        inline XPU T open_range_upper() {
            return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
        }

        template<>
        inline XPU uint16_t empty_bounds_lower<uint16_t>() { return uint16_t(USHRT_MAX); }

        template<>
        inline XPU uint16_t empty_bounds_upper<uint16_t>() { return uint16_t(0); }

        template<>
        inline XPU uint16_t open_range_lower<uint16_t>() { return uint16_t(0); }

        template<>
        inline XPU uint16_t open_range_upper<uint16_t>() { return uint16_t(USHRT_MAX); }

        template<>
        inline XPU int16_t empty_bounds_lower<int16_t>() { return int16_t(SHRT_MAX); }

        template<>
        inline XPU int16_t empty_bounds_upper<int16_t>() { return int16_t(SHRT_MIN); }

        template<>
        inline XPU int16_t open_range_lower<int16_t>() { return int16_t(SHRT_MIN); }

        template<>
        inline XPU int16_t open_range_upper<int16_t>() { return int16_t(SHRT_MAX); }

        template<>
        inline XPU uint8_t empty_bounds_lower<uint8_t>() { return uint8_t(CHAR_MAX); }

        template<>
        inline XPU uint8_t empty_bounds_upper<uint8_t>() { return uint8_t(CHAR_MIN); }

        template<>
        inline XPU uint8_t open_range_lower<uint8_t>() { return uint8_t(CHAR_MIN); }

        template<>
        inline XPU uint8_t open_range_upper<uint8_t>() { return uint8_t(CHAR_MAX); }

        template<>
        inline XPU int8_t empty_bounds_lower<int8_t>() { return int8_t(SCHAR_MIN); }

        template<>
        inline XPU int8_t empty_bounds_upper<int8_t>() { return int8_t(SCHAR_MAX); }

        template<>
        inline XPU int8_t open_range_lower<int8_t>() { return int8_t(SCHAR_MAX); }

        template<>
        inline XPU int8_t open_range_upper<int8_t>() { return int8_t(SCHAR_MIN); }


    } // luminous::constant
} // luminous