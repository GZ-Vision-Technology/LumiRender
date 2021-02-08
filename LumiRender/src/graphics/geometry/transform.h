//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    inline namespace geometry {

        struct Transform {
        private:
            float4x4 _mat;
            float4x4 _inv_mat;

        public:
            XPU Transform(float4x4 mat = float4x4(1))
                    : _mat(mat),
                      _inv_mat(::luminous::inverse(mat)) {}

            XPU Transform(float4x4 mat, float4x4 inv)
                    : _mat(mat),
                      _inv_mat(inv) {}

            XPU auto mat4x4() const {
                return _mat;
            }

            XPU auto mat3x3() const {
                return make_float3x3(_mat);
            }

            XPU auto inv_mat3x3() const {
                return luminous::inverse(mat3x3());
            }

            XPU float3 apply_point(float3 point) {
                float4 homo_point = make_float4(point, 1.f);
                homo_point = _mat * homo_point;
                return make_float3(homo_point);
            }

            XPU float3 apply_vector(float3 vec) {
                return mat3x3() * vec;
            }

            XPU float3 apply_normal(float3 normal) {
                return transpose(inv_mat3x3()) * normal;
            }

            XPU Transform operator*(const Transform &t) const {
                return Transform(_mat * t.mat4x4());
            }

            XPU Transform inverse() const {
                return Transform(_inv_mat, _mat);
            }

            [[nodiscard]] std::string to_string() const {
                return string_printf("transform:%s", _mat.to_string().c_str());
            }

            XPU static Transform translation(float3 t) {
                auto mat = make_float4x4(
                        1.f, 0.f, 0.f, 0.f,
                        0.f, 1.f, 0.f, 0.f,
                        0.f, 0.f, 1.f, 0.f,
                        t.x, t.y, t.z, 1.f);
                auto inv = make_float4x4(
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        -t.x, -t.y, -t.z, 1.0f);
                return Transform(mat, inv);
            }

            XPU static Transform scale(float3 s) {
                auto mat = make_float4x4(
                        s.x, 0.f, 0.f, 0.f,
                        0.f, s.y, 0.f, 0.f,
                        0.f, 0.f, s.z, 0.f,
                        0.f, 0.f, 0.f, 1.f);
                auto inv = make_float4x4(
                        1 / s.x, 0.0f, 0.0f, 0.0f,
                        0.0f, 1 / s.y, 0.0f, 0.0f,
                        0.0f, 0.0f, 1 / s.z, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f);
                return Transform(mat, inv);
            }

            [[nodiscard]] XPU static Transform scale(float s) {
                return scale(make_float3(s));
            }

            XPU static Transform rotation(const float3 axis, float angle, bool radian = false) noexcept {
                angle = radian ? angle : radians(angle);

                auto c = cos(angle);
                auto s = sin(angle);
                auto a = normalize(axis);
                auto t = (1.0f - c) * a;

                auto mat = make_float4x4(
                        c + t.x * a.x, t.x * a.y + s * a.z, t.x * a.z - s * a.y, 0.0f,
                        t.y * a.x - s * a.z, c + t.y * a.y, t.y * a.z + s * a.x, 0.0f,
                        t.z * a.x + s * a.y, t.z * a.y - s * a.x, c + t.z * a.z, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f);

                return Transform(mat, transpose(mat));
            }

            XPU static Transform rotation_x(float angle, bool radian = false) noexcept {
                return rotation(make_float3(1, 0, 0), angle, radian);
            }

            XPU static Transform rotation_y(float angle, bool radian = false) noexcept {
                return rotation(make_float3(0, 1, 0), angle, radian);
            }

            XPU static Transform rotation_z(float angle, bool radian = false) noexcept {
                return rotation(make_float3(0, 0, 1), angle, radian);
            }
        };


        /**
          * 跟复数一样，四元数用来表示旋转
          * q = w + xi + yj + zk
          * 其中 i^2 = j^2 = k^2 = ijk = -1
          * 实部为w，虚部为x,y,z
          * 单位四元数为 x^2 + y^2 + z^2 + w^2 = 1

          * 四元数的乘法法则与复数相似
          * qq' = (qw + qxi + qyj + qxk) * (q'w + q'xi + q'yj + q'xk)
          * 展开整理之后
          * (qq')xyz = cross(qxyz, q'xyz) + qw * q'xyz + q'w * qxyz
          * (qq')w = qw * q'w - dot(qxyz, q'xyz)

          * 四元数用法
          * 一个点p在绕某个向量单位v旋转2θ之后p',其中旋转四元数为q = (cosθ, v * sinθ)，q为单位四元数
          * 则满足p' = q * p * p^-1
          */
        XPU [[nodiscard]] static Quaternion matrix_to_quaternion(const float4x4 &m) {
            float x, y, z, w;
            float trace = m[0][0] + m[1][1] + m[2][2];
            if (trace > 0.f) {
                // Compute w from matrix trace, then xyz
                // 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
                float s = std::sqrt(trace + 1.0f);
                w = s / 2.0f;
                s = 0.5f / s;
                x = (m[2][1] - m[1][2]) * s;
                y = (m[0][2] - m[2][0]) * s;
                z = (m[1][0] - m[0][1]) * s;
            } else {
                // Compute largest of $x$, $y$, or $z$, then remaining components
                const int nxt[3] = {1, 2, 0};
                float q[3];
                int i = 0;
                if (m[1][1] > m[0][0]) i = 1;
                if (m[2][2] > m[i][i]) i = 2;
                int j = nxt[i];
                int k = nxt[j];
                float s = std::sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0f);
                q[i] = s * 0.5f;
                if (s != 0.f) s = 0.5f / s;
                w = (m[k][j] - m[j][k]) * s;
                q[j] = (m[j][i] + m[i][j]) * s;
                q[k] = (m[k][i] + m[i][k]) * s;
                x = q[0];
                y = q[1];
                z = q[2];
            }
            return Quaternion(make_float3(x, y, z), w);
        }

        XPU [[nodiscard]] static float4x4 quaternion_to_matrix(const Quaternion &q) noexcept {
            float x = q.v.x;
            float y = q.v.y;
            float z = q.v.z;
            float w = q.w;
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, xz = x * z, yz = y * z;
            float wx = x * w, wy = y * w, wz = z * w;
            auto ret = make_float4x4(1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy),     0,
                                     2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx),     0,
                                     2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy), 0,
                                     0,                 0,                 0,                 0);

            return transpose(ret);
        }

        XPU void decompose(const float4x4 &matrix, float3 *t, Quaternion *r, float3 *s) noexcept {

            auto M = matrix;
            for (int i = 0; i < 3; ++i) {
                M[i][3] = M[3][i] = 0.0f;
            }
            M[3][3] = 1.f;

            float norm = 0;
            int count = 0;
            auto R = M;
            do {
                // 计算下一个矩阵
                float4x4 R_next;
                float4x4 Rit = inverse(transpose(R));
                R_next = 0.5f * (R + Rit);

                // 对比两个矩阵的差异，如果差异小于0.0001则分解完成
                norm = 0;
                for (int i = 0; i < 3; ++i) {
                    float n = std::abs(R[i][0] - R_next[i][0]) +
                              std::abs(R[i][1] - R_next[i][1]) +
                              std::abs(R[i][2] - R_next[i][2]);
                    norm = std::max(norm, n);
                }
                R = R_next;
            } while (++count < 100 && norm > .0001);

            float4x4 S = inverse(R) * M;

            // extract translation component
            *t = make_float3(matrix[3]);
            *r = matrix_to_quaternion(R);
            *s = make_float3(S[0][0], S[1][1], S[2][2]);

        }
    }
}