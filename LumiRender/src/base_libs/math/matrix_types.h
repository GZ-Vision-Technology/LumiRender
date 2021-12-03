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

            LM_XPU explicit constexpr Matrix3x3(scalar_t s = 1) noexcept
                    : cols{vector_t(s, (scalar_t) 0, (scalar_t) 0),
                           vector_t((scalar_t) 0, s, (scalar_t) 0),
                           vector_t((scalar_t) 0, (scalar_t) 0, s)} {

            }

            LM_XPU explicit constexpr Matrix3x3(vector_t c0, vector_t c1, vector_t c2) noexcept
                    : cols{c0, c1, c2} {

            }

            LM_XPU explicit constexpr Matrix3x3(scalar_t m00, scalar_t m01, scalar_t m02,
                                                scalar_t m10, scalar_t m11, scalar_t m12,
                                                scalar_t m20, scalar_t m21, scalar_t m22) noexcept
                    : cols{vector_t(m00, m01, m02),
                           vector_t(m10, m11, m12),
                           vector_t(m20, m21, m22)} {

            }

            template<typename Index>
            LM_ND_XPU vector_t &operator[](Index i) noexcept {
                return cols[i];
            }

            template<typename Index>
            LM_ND_XPU constexpr vector_t operator[](Index i) const noexcept {
                return cols[i];
            }

            LM_XPU void print() const noexcept {
                printf("[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
                       cols[0].x, cols[0].y, cols[0].z,
                       cols[1].x, cols[1].y, cols[1].z,
                       cols[2].x, cols[2].y, cols[2].z);
            }

            GEN_STRING_FUNC({
                                return serialize("[", serialize(cols[0].to_string()), "\n",
                                                 serialize(cols[1].to_string()), "\n",
                                                 serialize(cols[2].to_string()), "]\n");
                            })
        };

        template<typename T>
        struct Matrix4x4 {
            using scalar_t = T;
            using vector_t = Vector<T, 4>;

            vector_t cols[4];

            LM_XPU explicit constexpr Matrix4x4(scalar_t s = 1) noexcept
                    : cols{vector_t(s, (scalar_t) 0, (scalar_t) 0, (scalar_t) 0),
                           vector_t((scalar_t) 0, s, (scalar_t) 0, (scalar_t) 0),
                           vector_t((scalar_t) 0, (scalar_t) 0, s, (scalar_t) 0),
                           vector_t((scalar_t) 0, (scalar_t) 0, (scalar_t) 0, s)} {}

            LM_XPU constexpr Matrix4x4(vector_t c0, vector_t c1, vector_t c2, vector_t c3) noexcept
                    : cols{c0, c1, c2, c3} {}

            LM_XPU constexpr Matrix4x4(scalar_t m00, scalar_t m01, scalar_t m02, scalar_t m03,
                                       scalar_t m10, scalar_t m11, scalar_t m12, scalar_t m13,
                                       scalar_t m20, scalar_t m21, scalar_t m22, scalar_t m23,
                                       scalar_t m30, scalar_t m31, scalar_t m32, scalar_t m33) noexcept
                    : cols{vector_t(m00, m01, m02, m03),
                           vector_t(m10, m11, m12, m13),
                           vector_t(m20, m21, m22, m23),
                           vector_t(m30, m31, m32, m33)} {}

            template<typename Index>
            LM_ND_XPU vector_t &operator[](Index i) noexcept {
                return cols[i];
            }

            template<typename Index>
            LM_ND_XPU constexpr vector_t operator[](Index i) const noexcept {
                return cols[i];
            }

            LM_XPU void print() const noexcept {
                printf("[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n",
                       cols[0].x, cols[0].y, cols[0].z, cols[0].w,
                       cols[1].x, cols[1].y, cols[1].z, cols[1].w,
                       cols[2].x, cols[2].y, cols[2].z, cols[2].w,
                       cols[3].x, cols[3].y, cols[3].z, cols[3].w);
            }

            GEN_STRING_FUNC({
                                return serialize("[", serialize(cols[0].to_string()), "\n",
                                                 serialize(cols[1].to_string()), "\n",
                                                 serialize(cols[2].to_string()), "\n",
                                                 serialize(cols[3].to_string()), "]\n");
                            })
        };

#define _define_matrix3x3(type)                                                                          \
        using type##3x3 = Matrix3x3<type>;                                                               \
        LM_ND_XPU constexpr auto make_##type##3x3(type val = 1) {                                        \
            return type##3x3(val);                                                                       \
        }                                                                                                \
        LM_ND_XPU constexpr auto make_##type##3x3(type##3x3::vector_t c0,                                \
                                                            type##3x3::vector_t  c1,                     \
                                                            type##3x3::vector_t  c2) noexcept {          \
            return type##3x3{c0, c1, c2};                                                                \
        }                                                                                                \
        LM_ND_XPU constexpr auto make_##type##3x3(                                                       \
                                    type m00, type m01, type m02,                                        \
                                    type m10, type m11, type m12,                                        \
                                    type m20, type m21, type m22) noexcept {                             \
            return type##3x3{m00, m01, m02, m10, m11, m12, m20, m21, m22};                               \
        }                                                                                                \
        LM_ND_XPU constexpr auto make_##type##3x3(const Matrix4x4<type> m) noexcept {                    \
            return type##3x3(make_##type##3(m[0]),make_##type##3(m[1]),make_##type##3(m[2]));            \
        }

#define _define_matrix4x4(type)                                                                         \
        using type##4x4 = Matrix4x4<type>;                                                              \
        LM_ND_XPU constexpr auto make_##type##4x4(type val = 1) noexcept {                              \
            return type##4x4{val};                                                                      \
        }                                                                                               \
        LM_ND_XPU constexpr auto make_##type##4x4(type##4x4::vector_t c0,                               \
                                        type##4x4::vector_t c1,                                         \
                                        type##4x4::vector_t c2,                                         \
                                        type##4x4::vector_t c3) noexcept {                              \
            return type##4x4{c0, c1, c2, c3};                                                           \
        }                                                                                               \
        LM_ND_XPU constexpr auto make_##type##4x4(                                                      \
                type m00, type m01, type m02, type m03,                                                 \
                type m10, type m11, type m12, type m13,                                                 \
                type m20, type m21, type m22, type m23,                                                 \
                type m30, type m31, type m32, type m33) noexcept {                                      \
            return type##4x4{m00, m01, m02, m03,                                                        \
                m10, m11, m12, m13,                                                                     \
                m20, m21, m22, m23,                                                                     \
                m30, m31, m32, m33};                                                                    \
        }                                                                                               \
        LM_ND_XPU constexpr auto make_##type##4x4(const type##3x3 m) noexcept {                         \
            return make_##type##4x4(                                                                    \
                    make_##type##4(m[0], (type)0),                                                      \
                    make_##type##4(m[1], (type)0),                                                      \
                    make_##type##4(m[2], (type)0),                                                      \
                    make_##type##4((type)0, (type)0, (type)0, (type)1));                                \
        }

#define define_matrix(type)                    \
        _define_matrix3x3(type)                \
        _define_matrix4x4(type)

        define_matrix(float)

        define_matrix(double)

#undef _define_matrix
#undef _define_matrix3x3
#undef _define_matrix4x4

        template<typename T>
        LM_ND_XPU constexpr Vector<T, 3> operator*(const Matrix3x3<T> m, Vector<T, 3> v) noexcept {
            return v.x * m[0] + v.y * m[1] + v.z * m[2];
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix4x4<T> operator*(const Matrix4x4<T> lhs, T v) noexcept {
            return Matrix4x4<T>(v * lhs[0], v * lhs[1], v * lhs[2], v * lhs[3]);
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix4x4<T> operator*(T v, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(v * rhs[0], v * rhs[1], v * rhs[2], v * rhs[3]);
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix3x3<T> operator*(const Matrix3x3<T> lhs, T v) noexcept {
            return Matrix3x3<T>(v * lhs[0], v * lhs[1], v * lhs[2]);
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix3x3<T> operator*(T v, const Matrix3x3<T> rhs) noexcept {
            return Matrix3x3<T>(v * rhs[0], v * rhs[1], v * rhs[2]);
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix3x3<T> operator*(const Matrix3x3<T> lhs, const Matrix3x3<T> rhs) noexcept {
            return Matrix3x3<T>(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]);
        }

        template<typename T>
        LM_ND_XPU constexpr Vector<T, 4> operator*(const Matrix4x4<T> m, Vector<T, 4> v) noexcept {
            return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix4x4<T> operator*(const Matrix4x4<T> lhs, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]);
        }

        template<typename T>
        LM_ND_XPU constexpr Matrix4x4<T> operator+(const Matrix4x4<T> lhs, const Matrix4x4<T> rhs) noexcept {
            return Matrix4x4<T>(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]);
        }
#define MAKE_VECTOR_UNARY_FUNC_BOOL(func)                                      \
    template<typename T, uint N>                                               \
    LM_ND_XPU constexpr auto is_##func##_v(Vector<T, N> v) noexcept {          \
        static_assert(N == 2 || N == 3 || N == 4);                             \
        if constexpr (N == 2) {                                                \
            return Vector<bool, 2>{is_##func(v.x), is_##func(v.y)};            \
        } else if constexpr (N == 3) {                                         \
            return Vector<bool, 3>(is_##func(v.x), is_##func(v.y),             \
                                    is_##func(v.z));                           \
        } else {                                                               \
            return Vector<bool, 4>(is_##func(v.x), is_##func(v.y),             \
                                    is_##func(v.z), is_##func(v.w));           \
        }                                                                      \
    }                                                                          \
    template<typename T, uint32_t N>                                           \
    ND_XPU_INLINE bool has_##func(Vector<T, N> v) noexcept {                   \
        return any(is_##func##_v(v));                                          \
    }                                                                          \
    template<typename T>                                                       \
    ND_XPU_INLINE bool has_##func(Matrix3x3<T> mat) noexcept {                 \
        return has_##func(mat.cols[0]) || has_##func(mat.cols[1])              \
            || has_##func(mat.cols[2]);                                        \
    }                                                                          \
    template<typename T>                                                       \
    ND_XPU_INLINE bool has_##func(Matrix4x4<T> mat) noexcept {                 \
        return has_##func(mat.cols[0]) || has_##func(mat.cols[1])              \
            || has_##func(mat.cols[2]) || has_##func(mat.cols[3]);             \
    }

        MAKE_VECTOR_UNARY_FUNC_BOOL(inf)
        MAKE_VECTOR_UNARY_FUNC_BOOL(nan)
        MAKE_VECTOR_UNARY_FUNC_BOOL(invalid)

#undef MAKE_VECTOR_UNARY_FUNC_BOOL
    }
}