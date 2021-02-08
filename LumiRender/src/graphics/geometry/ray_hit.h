//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace geometry {
        struct alignas(16) Ray {
        private:
            float _origin_x;
            float _origin_y;
            float _origin_z;
            float _t_min;
            float _direction_x;
            float _direction_y;
            float _direction_z;
            float _t_max;
        public:
            XPU Ray(const float3 origin, const float3 direction,
                    float t_max = luminous::constant::PosInfTy(),
                    float t_min = 0) noexcept:
                    _t_min(t_min),
                    _t_max(t_max) {
                update_origin(origin);
                update_direction(direction);
            }

            XPU void update_origin(const float3 origin) noexcept {
                _origin_x = origin.x;
                _origin_y = origin.y;
                _origin_z = origin.z;
            }

            XPU void update_direction(const float3 direction) noexcept {
                _direction_x = direction.x;
                _direction_y = direction.y;
                _direction_z = direction.z;
            }

            XPU [[nodiscard]] float3 origin() const noexcept {
                return make_float3(_origin_x, _origin_y, _origin_z);
            }

            XPU [[nodiscard]] float3 direction() const noexcept {
                return make_float3(_direction_x, _direction_y, _direction_z);
            }

            [[nodiscard]] std::string to_string() const {
                return string_printf("ray:{origin:%s,direction:%s,tmin:%f,tmax:%f}",
                                     origin().to_string().c_str(),
                                     direction().to_string().c_str(),
                                     _t_min,_t_max);
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