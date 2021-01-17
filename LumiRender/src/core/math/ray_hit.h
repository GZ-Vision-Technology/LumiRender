//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "math_util.h"
#include "data_types.h"


namespace luminous {
    inline namespace math {
        struct alignas(16) Ray {
            float origin_x;
            float origin_y;
            float origin_z;
            float t_min;
            float direction_x;
            float direction_y;
            float direction_z;
            float t_max;

            Ray(const float3 origin, const float3 direction,
                    float t_max = math::constant::float_infinity,
                    float t_min = 0) noexcept:
            t_min(t_min),
            t_max(t_max) {
                update_origin(origin);
                update_direction(direction);
            }

            XPU void update_origin(const float3 origin) noexcept {
                origin_x = origin.x;
                origin_y = origin.y;
                origin_z = origin.z;
            }

            XPU void update_direction(const float3 direction) noexcept {
                direction_x = direction.x;
                direction_y = direction.y;
                direction_z = direction.z;
            }
        };

        inline float3 offset_ray_origin(const float3 &p_in, const float3 &n_in) noexcept {

            constexpr auto origin = 1.0f / 32.0f;
            constexpr auto float_scale = 1.0f / 65536.0f;
            constexpr auto int_scale = 256.0f;

            float3 n = n_in;
            auto of_i = make_int3(static_cast<int>(int_scale * n.x),
                                  static_cast<int>(int_scale * n.y),
                                  static_cast<int>(int_scale * n.z));

            float3 p = p_in;
            float3 p_i = make_float3(
                    bits_to_float(bits_to_int(p.x) + select(p.x < 0, -of_i.x, of_i.x)),
                    bits_to_float(bits_to_int(p.y) + select(p.y < 0, -of_i.y, of_i.y)),
                    bits_to_float(bits_to_int(p.z) + select(p.z < 0, -of_i.z, of_i.z)));

            return select(abs(p) < origin, p + float_scale * n, p_i);
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