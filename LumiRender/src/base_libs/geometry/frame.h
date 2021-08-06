//
// Created by Zero on 2021/2/5.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace geometry {

        template<typename T>
        struct TFrame {
            using scalar_t = T;
            using vector_t = Vector<T, 3>;

            vector_t z;
            vector_t x, y;

            XPU TFrame() : x(T(1), T(0), T(0)), y(T(0), T(1), T(0)), z(T(0), T(0), T(1)) {}

            XPU TFrame(vector_t x, vector_t y, vector_t z) : x(x), y(y), z(z) {}

            XPU TFrame(vector_t normal)
                    : z(normal) {
                coordinateSystem(z, &x, &y);
            }

            NDSC_XPU static TFrame from_xy(vector_t x, vector_t y) {
                return TFrame(x, y, cross(x, y));
            }

            NDSC_XPU static TFrame from_xz(vector_t x, vector_t z) {
                return TFrame(x, cross(x, z), z);
            }

            NDSC_XPU static TFrame from_z(vector_t z) {
                return TFrame(z);
            }

            XPU void init(vector_t normal) {
                z = normal;
                coordinateSystem(z, &x, &y);
            }

            XPU vector_t to_local(vector_t world_v) const {
                return vector_t(dot(world_v, x), dot(world_v, y), dot(world_v, z));
            }

            XPU vector_t to_world(vector_t local_v) const {
                return x * local_v.x + y * local_v.y + z * local_v.z;
            }

            XPU static scalar_t cos_theta_2(vector_t v) {
                return sqr(v.z);
            }

            XPU static scalar_t cos_theta(const vector_t v) {
                return v.z;
            }

            XPU static scalar_t abs_cos_theta(vector_t v) {
                return std::abs(v.z);
            }

            XPU static scalar_t sin_theta_2(const vector_t v) {
                return 1.0f - cos_theta_2(v);
            }

            XPU static scalar_t sin_theta(const vector_t v) {
                scalar_t temp = sin_theta_2(v);
                if (temp <= 0.0f) {
                    return 0.0f;
                }
                return sqrt(temp);
            }

            XPU static scalar_t tan_theta(const vector_t &v) {
                scalar_t sin_theta2 = 1 - cos_theta_2(v);
                if (sin_theta2 <= 0.0f) {
                    return 0.0f;
                }
                return std::sqrt(sin_theta2) / cos_theta(v);
            }

            XPU static scalar_t tan_theta_2(const vector_t &v) {
                scalar_t cos_theta2 = cos_theta_2(v);
                scalar_t sin_theta2 = 1 - cos_theta2;
                if (sin_theta2 <= 0.0f) {
                    return 0.0f;
                }
                return sin_theta2 / cos_theta2;
            }

            XPU static scalar_t sin_phi(const vector_t &v) {
                scalar_t sinTheta = sin_theta(v);
                if (sinTheta == (scalar_t) 0) {
                    return 1;
                }
                return clamp(v.y / sinTheta, (scalar_t) -1, (scalar_t) 1);
            }

            XPU static scalar_t cos_phi(const vector_t &v) {
                scalar_t sinTheta = sin_theta(v);
                if (sinTheta == (scalar_t) 0) {
                    return 1;
                }
                return clamp(v.x / sinTheta, (scalar_t) -1, (scalar_t) 1);
            }

            XPU static scalar_t sin_phi_2(const vector_t &v) {
                return clamp(sqr(v.y) / sin_theta_2(v), (scalar_t) 0, (scalar_t) 1);
            }

            XPU static scalar_t cos_phi_2(const vector_t &v) {
                return clamp(sqr(v.x) / sin_theta_2(v), (scalar_t) 0, (scalar_t) 1);
            }

            XPU bool operator==(const TFrame &frame) const {
                return frame.x == x && frame.y == y && frame.z == z;
            }

            XPU bool operator!=(const TFrame &frame) const {
                return !operator==(frame);
            }

            XPU bool has_nan() const {
                return luminous::has_nan(x) || luminous::has_nan(y) || luminous::has_nan(z);
            }

            XPU bool has_inf() const {
                return luminous::has_inf(x) || luminous::has_inf(y) || luminous::has_inf(z);
            }

            GEN_STRING_FUNC({
                return string_printf("frame : {x:%s,y:%s,z:%s}",
                                     x.to_string().c_str(),
                                     y.to_string().c_str(),
                                     z.to_string().c_str());
            })
        };

        using Frame = TFrame<float>;

    } // luminous::geometry
} // luminous