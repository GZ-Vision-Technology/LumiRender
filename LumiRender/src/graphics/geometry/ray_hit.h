//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace geometry {
        struct alignas(16) Ray {
        public:
            float org_x{0.f};
            float org_y{0.f};
            float org_z{0.f};
            float t_min{0.f};
            float dir_x{0.f};
            float dir_y{0.f};
            float dir_z{0.f};
            float t_max{0.f};
        public:
            explicit XPU Ray(float t_max = ray_t_max,
                             float t_min = 0) noexcept: t_min(t_min),
                                                        t_max(t_max) {}

            XPU Ray(const float3 origin, const float3 direction,
                    float t_max = ray_t_max,
                    float t_min = 0) noexcept:
                    t_min(t_min),
                    t_max(t_max) {
                update_origin(origin);
                update_direction(direction);
            }

            XPU void update_origin(const float3 origin) noexcept {
                org_x = origin.x;
                org_y = origin.y;
                org_z = origin.z;
            }

            XPU void update_direction(const float3 direction) noexcept {
                dir_x = direction.x;
                dir_y = direction.y;
                dir_z = direction.z;
            }

            NDSC_XPU float3 origin() const noexcept {
                return make_float3(org_x, org_y, org_z);
            }

            NDSC_XPU float3 direction() const noexcept {
                return make_float3(dir_x, dir_y, dir_z);
            }

            NDSC_XPU bool has_nan() const noexcept {
                return origin().has_nan() || direction().has_nan();
            }

            NDSC_XPU bool has_inf() const noexcept {
                return origin().has_inf() || direction().has_inf();
            }

            XPU void print() const noexcept {
                printf("origin:[%f,%f,%f],direction[%f,%f,%f]\n",
                       org_x, org_y, org_z, dir_x, dir_y, dir_z);
            }

            [[nodiscard]] std::string to_string() const {
                return string_printf("ray:{origin:%s,direction:%s,tmin:%f,tmax:%f}",
                                     origin().to_string().c_str(),
                                     direction().to_string().c_str(),
                                     t_min, t_max);
            }
        };

        XPU inline float3 offset_ray_origin(const float3 &p_in, const float3 &n_in) noexcept {

            constexpr auto origin = 1.0f / 32.0f;
            constexpr auto float_scale = 1.0f / 65536.0f;
            constexpr auto int_scale = 256.0f;

            float3 n = n_in;
            auto of_i = make_int3(static_cast<int>(int_scale * n.x),
                                  static_cast<int>(int_scale * n.y),
                                  static_cast<int>(int_scale * n.z));

            float3 p = p_in;
            float3 p_i = make_float3(
                    bit_cast<float>(bit_cast<int>(p.x) + select(p.x < 0, -of_i.x, of_i.x)),
                    bit_cast<float>(bit_cast<int>(p.y) + select(p.y < 0, -of_i.y, of_i.y)),
                    bit_cast<float>(bit_cast<int>(p.z) + select(p.z < 0, -of_i.z, of_i.z)));

            return select(functor::abs(p) < origin, p + float_scale * n, p_i);
        }

        struct alignas(8) ClosestHit {
            float distance;
            uint triangle_id;
            uint instance_id;
            float2 bary;
        };

        struct AnyHit {
            float distance;
        };
    }
}