//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "thin_lens_camera.h"
#include "core/refl/factory.h"
#include "base_libs/sampling/warp.h"

namespace luminous {
    inline namespace render {
        ThinLensCamera::ThinLensCamera(const float4x4 &m, float fov_y, float velocity,
                                       float lens_radius, float focal_distance)
                : CameraBase(m, fov_y, velocity),
                  _lens_radius(lens_radius),
                  _focal_distance(focal_distance) {}

        // float ThinLensCamera::generate_ray(const SensorSample &ss, Ray *ray) {
        //     float3 p_film = make_float3(ss.p_film, 0);
        //     float3 p_sensor = _raster_to_camera.apply_point(p_film);
        //     auto c2w = camera_to_world();

        //     *ray = {make_float3(0, 0, 0), normalize(p_sensor)};
        //     if (_lens_radius > 0) {
        //         float2 p_lens = _lens_radius * square_to_disk(ss.p_lens);

        //         // Compute point on plane of focus
        //         float ft = _focal_distance / ray->dir_z;
        //         float3 p_focus = ray->at(ft);

        //         ray->update_origin(make_float3(p_lens, 0));
        //         ray->update_direction(normalize(p_focus - ray->origin()));
        //     }
        //     *ray = c2w.apply_ray(*ray);
        //     return 1 * ss.filter_weight;
        // }

        float ThinLensCamera::generate_ray(const SensorSample &ss, Ray *ray) {
            uint2 res = resolution();
            float2 p = (ss.p_film * 2.0f - float2(res)) * (std::tan(radians(_fov_y) * 0.5f) / res.y);
            auto direction = normalize(p.x * make_float3(camera_to_world().mat4x4()[0]) - p.y * make_float3(camera_to_world().mat4x4()[1]) + make_float3(camera_to_world().mat4x4()[2]));
            p.y *= -1;
            direction = camera_to_world_rotation().apply_vector(make_float3(p, 1));
            *ray = {position(), direction};
            return 1 * ss.filter_weight;
        }
    }
}