//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "../header.h"

namespace luminous {
    inline namespace geometry {

        struct TriangleHandle {
            index_t i{};
            index_t j{};
            index_t k{};


            TriangleHandle() = default;

            TriangleHandle(index_t i, index_t j, index_t k)
                    : i(i), j(j), k(k) {}

            LM_XPU void print() const {
                printf("i:%u, j:%u, k:%u \n", i, j, k);
            }
        };

        ND_XPU_INLINE bool same_hemisphere(float3 w1, float3 w2) {
            return w1.z * w2.z > 0;
        }

        ND_XPU_INLINE bool same_hemisphere(float3 w1, float3 w2, float3 n) {
            return dot(w1, n) * dot(w2, n) > 0;
        }

        LM_XPU inline float3 offset_ray_origin(const float3 &p_in, const float3 &n_in) noexcept {

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

        ND_XPU_INLINE float3 spherical_direction(float sin_theta, float cos_theta, float sin_phi, float cos_phi) {
            return make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        }

        ND_XPU_INLINE float3 spherical_direction(float sin_theta, float cos_theta, float phi) {
            return make_float3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
        }

        ND_XPU_INLINE float3 spherical_direction(float theta, float phi) {
            return spherical_direction(std::sin(theta), std::cos(theta), phi);
        }

        ND_XPU_INLINE float spherical_theta(float3 v) {
            return safe_acos(v.z);
        }

        ND_XPU_INLINE float spherical_phi(float3 v) {
            float p = std::atan2(v.y, v.x);
            return (p < 0) ? (p + 2 * Pi) : p;
        }

        struct alignas(16) Ray {
        public:
            float org_x{0.f};
            float org_y{0.f};
            float org_z{0.f};
            float dir_x{0.f};
            float dir_y{0.f};
            float dir_z{0.f};
            float t_max{0.f};
        public:
            explicit LM_XPU Ray(float t_max = ray_t_max) noexcept:
                    t_max(t_max) {}

            LM_XPU Ray(const float3 origin, const float3 direction,
                       float t_max = ray_t_max) noexcept:
                    t_max(t_max) {
                update_origin(origin);
                update_direction(direction);
            }

            LM_ND_XPU float3 at(float t) const {
                return origin() + direction() * t;
            }

            LM_XPU void update_origin(const float3 origin) noexcept {
                org_x = origin.x;
                org_y = origin.y;
                org_z = origin.z;
            }

            LM_XPU void update_direction(const float3 direction) noexcept {
                dir_x = direction.x;
                dir_y = direction.y;
                dir_z = direction.z;
            }

            LM_ND_XPU float3 origin() const noexcept {
                return make_float3(org_x, org_y, org_z);
            }

            LM_ND_XPU float3 direction() const noexcept {
                return make_float3(dir_x, dir_y, dir_z);
            }

            LM_ND_XPU bool has_nan() const noexcept {
                return luminous::has_nan(origin()) || luminous::has_nan(direction());
            }

            LM_ND_XPU bool has_inf() const noexcept {
                return luminous::has_inf(origin()) || luminous::has_inf(direction());
            }

            LM_XPU void print() const noexcept {
                printf("origin:[%f,%f,%f],direction[%f,%f,%f]\n",
                       org_x, org_y, org_z, dir_x, dir_y, dir_z);
            }

            LM_ND_XPU static Ray spawn_ray(float3 pos, float3 normal, float3 dir) {
                normal *= dot(normal, dir) > 0 ? 1 : -1;
                float3 org = offset_ray_origin(pos, normal);
                return Ray(org, dir);
            }

            LM_ND_XPU static Ray spawn_ray_to(float3 p_start, float3 n_start, float3 p_target) {
                float3 dir = p_target - p_start;
                float3 org = offset_ray_origin(p_start, n_start);
                n_start *= dot(n_start, dir) > 0 ? 1 : -1;
                return Ray(org, dir, 1 - shadow_epsilon);
            }

            LM_ND_XPU static Ray spawn_ray_to(float3 p_start, float3 n_start, float3 p_target, float3 n_target) {
                float3 dir = p_target - p_start;
                n_target *= dot(n_target, -dir) > 0 ? 1 : -1;
                p_target = offset_ray_origin(p_target, n_target);
                n_start *= dot(n_start, dir) > 0 ? 1 : -1;
                float3 org = offset_ray_origin(p_start, n_start);
                return Ray(org, dir, 1 - shadow_epsilon);
            }

            GEN_STRING_FUNC({
                                return string_printf("ray:{origin:%s,direction:%s,t_max:%f}",
                                                     origin().to_string().c_str(),
                                                     direction().to_string().c_str(),
                                                     t_max);
                            })
        };

        struct alignas(8) HitInfo {
            index_t instance_id{invalid_uint32};
            index_t prim_id{invalid_uint32};
            float2 bary{};

            LM_ND_XPU bool is_hit() const {
                return prim_id != invalid_uint32;
            }

            LM_XPU void print() const {
                printf("instance_id:%u,primitive id:%u, bary:(%f,%f)\n",
                       instance_id, prim_id, bary.x, bary.y);
            }
        };
    }
}