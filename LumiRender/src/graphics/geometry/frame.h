//
// Created by Zero on 2021/2/5.
//


#pragma once

namespace luminous {
    inline namespace geometry {

        template<typename T>
        struct TFrame {
            using scalar_t = T;
            using vector_t = Vector<T, 3>;

            vector_t n;
            vector_t s, t;

            XPU TFrame(vector_t normal)
                    : n(normal) {
                coordinateSystem(n, &s, &t);
            }

            XPU vector_t to_local(vector_t world_v) const {
                return vector_t(dot(world_v, s), dot(world_v, t), dot(world_v, n));
            }

            XPU vector_t to_world(vector_t local_v) const {
                return s * local_v.x + t * local_v.y + n * local_v.z;
            }

            XPU static scalar_t cos_theta_2(vector_t v) {
                return sqr(v.z);
            }

            XPU static scalar_t cos_theta(const vector_t v) {
                return v.z;
            }

            XPU static scalar_t sin_theta_2(const vector_t v) {
                return 1.0f - cos_theta_2(v);
            }

            XPU static scalar_t sin_theta(const vector_t v) {
                scalar_t temp = sin_theta_2(v);
                if (temp <= 0.0f)
                    return 0.0f;
                return sqrt(temp);
            }

            XPU static scalar_t tan_theta(const vector_t &v) {
                scalar_t sin_theta2 = 1 - cos_theta_2(v);
                if (sin_theta2 <= 0.0f)
                    return 0.0f;
                return std::sqrt(sin_theta2) / cos_theta(v);
            }

            XPU static scalar_t tan_theta_2(const vector_t &v) {
                scalar_t cos_theta2 = cos_theta_2(v);
                scalar_t sin_theta2 = 1 - cos_theta2;
                if (sin_theta2 <= 0.0f)
                    return 0.0f;
                return sin_theta2 / cos_theta2;
            }

            XPU static scalar_t sin_phi(const vector_t &v) {
                scalar_t sinTheta = sin_theta(v);
                if (sinTheta == (scalar_t) 0)
                    return 1;
                return clamp(v.y / sinTheta, (scalar_t) -1, (scalar_t) 1);
            }

            XPU static scalar_t cos_phi(const vector_t &v) {
                scalar_t sinTheta = sin_theta(v);
                if (sinTheta == (scalar_t) 0)
                    return 1;
                return clamp(v.x / sinTheta, (scalar_t) -1, (scalar_t) 1);
            }

            XPU static scalar_t sin_phi_2(const vector_t &v) {
                return clamp(sqr(v.y) / sin_theta_2(v), (scalar_t) 0, (scalar_t) 1);
            }

            XPU static scalar_t cos_phi_2(const vector_t &v) {
                return clamp(sqr(v.x) / sin_theta_2(v), (scalar_t) 0, (scalar_t) 1);
            }

            XPU bool operator==(const TFrame &frame) const {
                return frame.s == s && frame.t == t && frame.n == n;
            }

            XPU bool operator!=(const TFrame &frame) const {
                return !operator==(frame);
            }

            XPU bool has_nan() const {
                return s.has_nan() || t.has_nan() || n.has_nan();
            }

            [[nodiscard]] std::string to_string() const {
                return string_printf("frame : {x:%s,y:%s,z:%s}",
                                     s.to_string().c_str(),
                                     t.to_string().c_str(),
                                     n.to_string().c_str());
            }
        };

        using Frame = TFrame<float>;

    } // luminous::geometry
} // luminous