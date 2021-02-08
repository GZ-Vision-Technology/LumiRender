//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    inline namespace geometry {

        template<typename T>
        struct TTransform {
        public:
            using scalar_t = T;
        private:
            Matrix4x4<T> _mat;
            Matrix4x4<T> _inv_mat;

        public:
            XPU TTransform(Matrix4x4<T> mat = Matrix4x4<T>(1))
                : _mat(mat),
                _inv_mat(inverse(mat)) {}

            XPU TTransform(Matrix4x4<T> mat, Matrix4x4<T> inv)
                : _mat(mat),
                _inv_mat(inv) {}

            XPU auto mat4x4() const {
                return _mat;
            }

            XPU auto mat3x3() const {
                return Matrix<T, 3>(_mat[0].vec3(), _mat[1].vec3(), _mat[2].vec3());
            }

            XPU auto inv_mat3x3() const {
                return inverse(mat3x3());
            }

            XPU Vector<T, 3> apply_point(Vector<T, 3> point) {
                Vector<T, 4> homo_point = Vector<T, 4>(point.x, point.y, point.z, (T)1);
                homo_point = _mat * homo_point;
                return Vector<T, 3>(homo_point.x, homo_point.y, homo_point.z);
            }

            XPU Vector<T, 3> apply_vector(Vector<T, 3> vec) {
                return mat3x3() * vec;
            }

            XPU Vector<T, 3> apply_normal(Vector<T, 3> normal) {
                return transpose(inv_mat3x3()) * normal;
            }

            TTransform operator*(const TTransform &t) const { 
                return TTransform(m * t.mat4x4()); 
            }

            XPU static TTransform translate(Vector<T, 3> t) const {
                // auto mat = 
            }
        };

        using Transform = TTransform<float>;
    }
}