//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "vector_types.h"

namespace luminous {

    inline namespace matrix {

        template<typename T>
        struct Matrix3x3 {
            using scalar_t = T;
            using vector_t = Vector<T, 3>;

            vector_t cols[3];

            XPU explicit constexpr Matrix3x3(scalar_t s = 1) noexcept
                : cols{vector_t(s, (scalar_t)0, (scalar_t)0),
                       vector_t((scalar_t)0, s, (scalar_t)0),
                       vector_t((scalar_t)0, (scalar_t)0, s)} {

            }

            XPU explicit constexpr Matrix3x3(vector_t c0, vector_t c1, vector_t c2) noexcept
                :cols{c0, c1, c2} {

            }

            XPU explicit constexpr Matrix3x3(scalar_t m00, scalar_t m01, scalar_t m02,
                                             scalar_t m10, scalar_t m11, scalar_t m12,
                                             scalar_t m20, scalar_t m21, scalar_t m22) noexcept
                    : cols{vector_t(m00, m01, m02),
                           vector_t(m10, m11, m12),
                           vector_t(m20, m21, m22)} {

            }

            template<typename Index>
            XPU [[nodiscard]] vector_t &operator[](Index i) noexcept {
                return cols[i];
            }

            template<typename Index>
            XPU [[nodiscard]] constexpr vector_t operator[](Index i) const noexcept {
                return cols[i];
            }

            XPU [[nodiscard]] bool has_nan() const noexcept {
                return cols[0].has_nan() || cols[1].has_nan() || cols[2].has_nan();
            }

            [[nodiscard]] std::string to_string() const {
                return serialize("[", serialize(cols[0].to_string()), "\n",
                                 serialize(cols[1].to_string()), "\n",
                                 serialize(cols[2].to_string()), "]\n");
            }
        };

        template<typename T>
        struct Matrix4x4 {
            using scalar_t = T;
            using vector_t = Vector<T, 4>;

            vector_t cols[4];

            XPU explicit constexpr Matrix4x4(scalar_t s = 1) noexcept
                    : cols{vector_t(s, (scalar_t)0, (scalar_t)0, (scalar_t)0),
                           vector_t((scalar_t)0, s, (scalar_t)0, (scalar_t)0),
                           vector_t((scalar_t)0, (scalar_t)0, s, (scalar_t)0),
                           vector_t((scalar_t)0, (scalar_t)0, (scalar_t)0, s)} {}

            XPU constexpr Matrix4x4(vector_t c0, vector_t c1, vector_t c2, vector_t c3) noexcept
                : cols{c0, c1, c2, c3} {}

            XPU constexpr Matrix4x4(scalar_t m00, scalar_t m01, scalar_t m02, scalar_t m03,
                                    scalar_t m10, scalar_t m11, scalar_t m12, scalar_t m13,
                                    scalar_t m20, scalar_t m21, scalar_t m22, scalar_t m23,
                                    scalar_t m30, scalar_t m31, scalar_t m32, scalar_t m33) noexcept
                    : cols{vector_t(m00, m01, m02, m03),
                           vector_t(m10, m11, m12, m13),
                           vector_t(m20, m21, m22, m23),
                           vector_t(m30, m31, m32, m33)} {}

            template<typename Index>
            XPU [[nodiscard]] vector_t &operator[](Index i) noexcept {
                return cols[i];
            }

            template<typename Index>
            XPU [[nodiscard]] constexpr vector_t operator[](Index i) const noexcept {
                return cols[i];
            }

            XPU [[nodiscard]] bool has_nan() const noexcept {
                return cols[0].has_nan() || cols[1].has_nan() || cols[2].has_nan() || cols[3].has_nan();
            }

            [[nodiscard]] std::string to_string() const {
                return serialize("[", serialize(cols[0].to_string()), "\n",
                                 serialize(cols[1].to_string()), "\n",
                                 serialize(cols[2].to_string()), "\n",
                                 serialize(cols[3].to_string()), "]\n");
            }
        };

#define _define_matrix3x3(type)                                                                          \
        using type##3x3 = Matrix3x3<type>;                                                               \
        XPU [[nodiscard]] constexpr auto make_##type##3x3(type val = 1) {                                \
            return type##3x3(val);                                                                       \
        }                                                                                                \
        XPU [[nodiscard]] constexpr auto make_##type##3x3(type##3x3::vector_t c0,                        \
                                                            type##3x3::vector_t  c1,                     \
                                                            type##3x3::vector_t  c2) noexcept {          \
            return type##3x3{c0, c1, c2};                                                                \
        }                                                                                                \
        XPU [[nodiscard]] constexpr auto make_##type##3x3(                                               \
                                    type m00, type m01, type m02,                                        \
                                    type m10, type m11, type m12,                                        \
                                    type m20, type m21, type m22) noexcept {                             \
            return type##3x3{m00, m01, m02, m10, m11, m12, m20, m21, m22};                               \
        }                                                                                                \
        XPU [[nodiscard]] constexpr auto make_##type##3x3(const Matrix4x4<type> m) noexcept {            \
            return type##3x3(make_##type##3(m[0]),make_##type##3(m[1]),make_##type##3(m[2]));            \
        }

#define _define_matrix4x4(type)                                                                         \
        using type##4x4 = Matrix4x4<type>;                                                              \
        XPU [[nodiscard]] constexpr auto make_##type##4x4(type val = 1) noexcept {                      \
            return type##4x4{val};                                                                      \
        }                                                                                               \
        XPU [[nodiscard]] constexpr auto make_##type##4x4(type##4x4::vector_t c0,                       \
                                        type##4x4::vector_t c1,                                         \
                                        type##4x4::vector_t c2,                                         \
                                        type##4x4::vector_t c3) noexcept {                              \
            return type##4x4{c0, c1, c2, c3};                                                           \
        }                                                                                               \
        XPU [[nodiscard]] constexpr auto make_##type##4x4(                                              \
                type m00, type m01, type m02, type m03,                                                 \
                type m10, type m11, type m12, type m13,                                                 \
                type m20, type m21, type m22, type m23,                                                 \
                type m30, type m31, type m32, type m33) noexcept {                                      \
            return type##4x4{m00, m01, m02, m03,                                                        \
                m10, m11, m12, m13,                                                                     \
                m20, m21, m22, m23,                                                                     \
                m30, m31, m32, m33};                                                                    \
        }                                                                                               \
        XPU [[nodiscard]] constexpr auto make_##type##4x4(const type##3x3 m) noexcept {                 \
            return make_##type##4x4(                                                                    \
                    make_##type##4(m[0], (type)0),                                                      \
                    make_##type##4(m[1], (type)0),                                                      \
                    make_##type##4(m[2], (type)0),                                                      \
                    make_##type##4((type)0, (type)0, (type)0, (type)1));                                \
        }

#define _define_matrix(type)                    \
        _define_matrix3x3(type)                 \
        _define_matrix4x4(type)

        _define_matrix(float)
        _define_matrix(double)

#undef _define_matrix
#undef _define_matrix3x3
#undef _define_matrix4x4

        template<typename T>
        XPU [[nodiscard]] constexpr Vector<T, 3> operator*(const Matrix3x3<T> m, Vector<T, 3> v) noexcept {
            return v.x * m[0] + v.y * m[1] + v.z * m[2];
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix4x4<T> operator*(const Matrix4x4<T> lhs, T v) noexcept {
            return Matrix4x4<T>(v * lhs[0], v * lhs[1], v * lhs[2], v * lhs[3]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix4x4<T> operator*(T v, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(v * rhs[0], v * rhs[1], v * rhs[2], v * rhs[3]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix3x3<T> operator*(const Matrix3x3<T> lhs, T v) noexcept {
            return Matrix3x3<T>(v * lhs[0], v * lhs[1], v * lhs[2]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix3x3<T> operator*(T v, const Matrix3x3<T> rhs) noexcept {
            return Matrix3x3<T>(v * rhs[0], v * rhs[1], v * rhs[2]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix3x3<T> operator*(const Matrix3x3<T> lhs, const Matrix3x3<T> rhs) noexcept {
            return Matrix3x3<T>(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Vector<T, 4> operator*(const Matrix4x4<T> m, Vector<T, 4> v) noexcept {
            return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix4x4<T> operator*(const Matrix4x4<T> lhs, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]);
        }

        template<typename T>
        XPU [[nodiscard]] constexpr Matrix4x4<T> operator+(const Matrix4x4<T> lhs, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]);
        }
    }
}