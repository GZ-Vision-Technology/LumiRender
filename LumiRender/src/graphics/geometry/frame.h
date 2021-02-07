//
// Created by Zero on 2021/2/5.
//


#pragma once

namespace luminous {
    inline namespace geometry {
        
        template <typename T>
        struct TFrame {
            using scalar_t = T;
            using vector_t = Vector<T, 3>;

            vector_t n;
            vector_t s,t;

            XPU TFrame(vector_t normal)
                :n(normal) {
                coordinateSystem(n, &s, &t);
            }

            XPU vector_t to_local(vector_t world_v) const {
                return vector_t(dot(world_v, s), dot(world_v, t), dot(world_v, n));
            }

            XPU vector_t to_world(vector_t local_v) const {
                return s * local_v.x + t * local_v.y + n * local_v.z;
            }

            XPU scalar_t cos_theta_2(vector_t &v) {
                return sqr(v.z);
            }

        };

        using Frame = TFrame<float>;

    } // luminous::geometry
} // luminous