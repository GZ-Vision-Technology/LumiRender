//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "data_types.h"
#include "math_util.h"

namespace luminous {
    struct alignas(16) Ray {
        float origin_x;
        float origin_y;
        float origin_z;
        float min_distance;
        float direction_x;
        float direction_y;
        float direction_z;
        float max_distance;
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


}