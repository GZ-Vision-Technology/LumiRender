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

            XPU Vector<T, 3> apply_point(Vector<T, 3> point) {

            }

            XPU Vector<T, 3> apply_vector(Vector<T, 3> point) {

            }

            XPU Vector<T, 3> apply_normal(Vector<T, 3> point) {

            }
        };

        using Transform = TTransform<float>;
    }
}