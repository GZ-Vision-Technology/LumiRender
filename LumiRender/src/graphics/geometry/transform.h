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
                _inv_mat(inverse(mat)) {}

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
                return inverse(mat3x3());
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
                        1/s.x, 0.0f, 0.0f, 0.0f,
                        0.0f, 1/s.y, 0.0f, 0.0f,
                        0.0f, 0.0f, 1/s.z, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f);
                return Transform(mat, inv);
            }
        };
    }
}