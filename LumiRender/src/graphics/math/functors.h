//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "matrix_types.h"

namespace luminous {
    inline namespace functor {
        using std::abs;

        // bit manipulation function
        [[nodiscard]] constexpr auto next_pow_of_two(uint v) noexcept {
            v--;
            v |= v >> 1u;
            v |= v >> 2u;
            v |= v >> 4u;
            v |= v >> 8u;
            v |= v >> 16u;
            v++;
            return v;
        }

        [[nodiscard]] constexpr float radians(float deg) noexcept {
            return deg * constant::Pi / 180.0f;
        }
        [[nodiscard]] constexpr float degrees(float rad) noexcept {
            return rad * constant::invPi * 180.0f;
        }

        template<typename T, typename F>
        [[nodiscard]] constexpr auto select(bool pred, T t, F f) noexcept {
            return pred ? t : f;
        }

        template<typename T, uint N, std::enable_if_t<scalar::is_scalar<T>, int> = 0>
        [[nodiscard]] constexpr auto select(Vector<bool, N> pred, Vector<T, N> t, Vector<T, N> f) noexcept {
            static_assert(N == 2 || N == 3 || N == 4);
            if constexpr (N == 2) {
                return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y)};
            } else if constexpr (N == 3) {
                return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z)};
            } else {
                return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z), select(pred.w, t.w, f.w)};
            }
        }

        template<typename A, typename B>
        [[nodiscard]] constexpr auto lerp(A a, B b, float t) noexcept {
            return a + t * (b - a);
        }

        template <typename T, typename U, typename V>
        [[nodiscard]] constexpr T clamp(T val, U low, V high) noexcept {
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
        inline void sincos(float theta, float *_sin, float *_cos) {
            *_sin = sinf(theta);
            *_cos = cosf(theta);
        }
#endif

        inline float safe_sqrt(float x) noexcept {
            return sqrt(std::max(x, 0.f));
        }

        inline float safe_acos(float x) noexcept {
            return acos(clamp(x, -1.f, 1.f));
        }

        inline float safe_asin(float x) noexcept {
            return asin(clamp(x, -1.f, 1.f));
        }

        template <typename T>
        inline constexpr auto sqr(T v) {
            return v * v;
        }

        inline XPU float rcp(float f) noexcept { return 1.f/f; }

        inline XPU double rcp(double d) noexcept { return 1./d; }

        XPU float saturate(const float &f) { return std::min(1.f,std::max(0.f,f)); }

        template <int n>
        inline constexpr float Pow(float v) {
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

        inline bool is_power_of_two(uint32_t i) noexcept { return (i & (i-1)) == 0; }

        inline bool is_power_of_two(int32_t i) noexcept { return i > 0 && (i & (i-1)) == 0; }

        inline bool is_power_of_two(uint64_t i) noexcept { return (i & (i-1)) == 0; }

        inline bool is_power_of_two(int64_t i) noexcept { return i > 0 && (i & (i-1)) == 0; }


        // Vector Functions
#define MAKE_VECTOR_UNARY_FUNC(func)                                          \
    template<typename T, uint N>                                              \
    [[nodiscard]] constexpr auto func(Vector<T, N> v) noexcept {              \
        static_assert(N == 2 || N == 3 || N == 4);                            \
        if constexpr (N == 2) {                                               \
            return Vector<T, 2>{func(v.x), func(v.y)};                        \
        } else if constexpr (N == 3) {                                        \
            return Vector<T, 3>(func(v.x), func(v.y), func(v.z));             \
        } else {                                                              \
            return Vector<T, 4>(func(v.x), func(v.y), func(v.z), func(v.w));  \
        }                                                                     \
    }
        MAKE_VECTOR_UNARY_FUNC(rcp)
        MAKE_VECTOR_UNARY_FUNC(saturate)
        MAKE_VECTOR_UNARY_FUNC(abs)

        template<typename T, uint N>
        [[nodiscard]] constexpr auto volume(Vector<T, N> v) noexcept {
            static_assert(N == 2 || N == 3 || N == 4);
            if constexpr (N == 2) {
                return v.x * v.y;
            } else if constexpr (N == 3) {
                return v.x * v.y * v.z;
            } else {
                return v.x * v.y * v.z * v.w;
            }
        }

        template<uint N>
        [[nodiscard]] constexpr auto dot(Vector<float, N> u, Vector<float, N> v) noexcept {
            static_assert(N == 2 || N == 3 || N == 4);
            if constexpr (N == 2) {
                return u.x * v.x + u.y * v.y;
            } else if constexpr (N == 3) {
                return u.x * v.x + u.y * v.y + u.z * v.z;
            } else {
                return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
            }
        }

        template<uint N>
        [[nodiscard]] constexpr auto length(Vector<float, N> u) noexcept {
            return sqrt(dot(u, u));
        }

        template<uint N>
        [[nodiscard]] constexpr auto length_squared(Vector<float, N> u) noexcept {
            return dot(u, u);
        }

        template<uint N>
        [[nodiscard]] constexpr auto normalize(Vector<float, N> u) noexcept {
            return u * (1.0f / length(u));
        }

        template<uint N>
        [[nodiscard]] constexpr auto distance(Vector<float, N> u, Vector<float, N> v) noexcept {
            return length(u - v);
        }

        template<uint N>
        [[nodiscard]] constexpr auto distance_squared(Vector<float, N> u, Vector<float, N> v) noexcept {
            return length_squared(u - v);
        }

        [[nodiscard]] constexpr auto cross(float3 u, float3 v) noexcept {
            return make_float3(u.y * v.z - v.y * u.z,
                               u.z * v.x - v.z * u.x,
                               u.x * v.y - v.x * u.y);
        }

        // Quaternion Functions
        [[nodiscard]] float dot(Quaternion q1, Quaternion q2) noexcept {
            return dot(q1.v, q2.v) + q1.w * q2.w;
        }

        [[nodiscard]] Quaternion normalize(Quaternion q) noexcept {
            return q / std::sqrt(dot(q, q));
        }

        [[nodiscard]] Quaternion slerp(float t, const Quaternion &q1, const Quaternion &q2) {
            float cosTheta = dot(q1, q2);
            if (cosTheta > .9995f)
                //如果旋转角度特别小，当做直线处理
                return normalize(q1 * (1 - t) + q2 * t);
            else {
                // 原始公式 result = (q1sin((1-t)θ) + q2sin(tθ)) / (sinθ)
                // 比较直观的理解qperp = q2 - cosθ * q1 = q2 - dot(q1, q2) * q1
                // q' = q1 * cos(θt) + qperp * sin(θt)
                float theta = std::acos(clamp(cosTheta, -1, 1));
                float theta_p = theta * t;
                Quaternion q_perp = normalize(q2 - q1 * cosTheta);
                return q1 * std::cos(theta_p) + q_perp * std::sin(theta_p);
            }
        }

        // Matrix Functions
        [[nodiscard]] constexpr auto transpose(float3x3 m) noexcept {
            return make_float3x3(m[0].x, m[1].x, m[2].x,
                                 m[0].y, m[1].y, m[2].y,
                                 m[0].z, m[1].z, m[2].z);
        }

        [[nodiscard]] constexpr auto transpose(float4x4 m) noexcept {
            return make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x,
                                 m[0].y, m[1].y, m[2].y, m[3].y,
                                 m[0].z, m[1].z, m[2].z, m[3].z,
                                 m[0].w, m[1].w, m[2].w, m[3].w);
        }

        [[nodiscard]] auto inverse(float3x3 m) noexcept {// from GLM
            auto one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
            return make_float3x3(
                    (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
                    (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
                    (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
                    (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
                    (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
                    (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
                    (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
                    (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
                    (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
        }

        [[nodiscard]] constexpr auto inverse(const float4x4 m) noexcept {// from GLM
            const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
            const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
            const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
            const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
            const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
            const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
            const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
            const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
            const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
            const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
            const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
            const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
            const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
            const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
            const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
            const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
            const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
            const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
            const auto fac0 = make_float4(coef00, coef00, coef02, coef03);
            const auto fac1 = make_float4(coef04, coef04, coef06, coef07);
            const auto fac2 = make_float4(coef08, coef08, coef10, coef11);
            const auto fac3 = make_float4(coef12, coef12, coef14, coef15);
            const auto fac4 = make_float4(coef16, coef16, coef18, coef19);
            const auto fac5 = make_float4(coef20, coef20, coef22, coef23);
            const auto Vec0 = make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
            const auto Vec1 = make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
            const auto Vec2 = make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
            const auto Vec3 = make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
            const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
            const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
            const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
            const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
            const auto sign_a = make_float4(+1, -1, +1, -1);
            const auto sign_b = make_float4(-1, +1, -1, +1);
            const auto inv_0 = inv0 * sign_a;
            const auto inv_1 = inv1 * sign_b;
            const auto inv_2 = inv2 * sign_a;
            const auto inv_3 = inv3 * sign_b;
            const auto dot0 = m[0] * make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
            const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
            const auto one_over_determinant = 1.0f / dot1;
            return make_float4x4(inv_0 * one_over_determinant,
                                 inv_1 * one_over_determinant,
                                 inv_2 * one_over_determinant,
                                 inv_3 * one_over_determinant);
        }

        // other functions
        template <class To, class From>
        XPU typename std::enable_if_t<sizeof(To) == sizeof(From) &&
                                               std::is_trivially_copyable_v<From> &&
                                               std::is_trivially_copyable_v<To>, To>
        bit_cast(const From &src) noexcept {
            static_assert(std::is_trivially_constructible_v<To>,
                          "This implementation requires the destination type to be trivially "
                          "constructible");
            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }

        XPU [[nodiscard]] inline float fast_exp(float x) {
    #ifdef IS_GPU_CODE
            return __expf(x);
    #else
            return std::exp(x);
    #endif
        }

        XPU [[nodiscard]] float gaussian(float x, float mu = 0, float sigma = 1) {
            return 1 / std::sqrt(2 * Pi * sigma * sigma) *
                   fast_exp(-sqr(x - mu) / (2 * sigma * sigma));
        }

        XPU [[nodiscard]] float gaussian_integral(float x0, float x1, float mu = 0,
                                       float sigma = 1) {
            assert(sigma > 0);
            float sigmaRoot2 = sigma * float(1.414213562373095);
            return 0.5f * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
        }


        template <typename Predicate>
        [[nodiscard]] XPU inline size_t find_interval(size_t sz, const Predicate &pred) {
            using ssize_t = std::make_signed_t<size_t>;
            ssize_t size = (ssize_t)sz - 2, first = 1;
            while (size > 0) {
                // Evaluate predicate at midpoint and update _first_ and _size_
                size_t half = (size_t)size >> 1, middle = first + half;
                bool pred_result = pred(middle);
                first = pred_result ? middle + 1 : first;
                size = pred_result ? size - (half + 1) : half;
            }
            return (size_t)clamp((ssize_t)first - 1, 0, sz - 2);
        }

    } // luminous::functors
} // luminous