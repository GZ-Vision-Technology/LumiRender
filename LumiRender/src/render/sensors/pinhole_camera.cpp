//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "pinhole_camera.h"

namespace luminous {
    inline namespace render {

        PinholeCamera::PinholeCamera(const float4x4 m, float fov_y, float velocity)
                : CameraBase(m, fov_y, velocity) {}

        float PinholeCamera::generate_ray(const SensorSample &ss, Ray *ray) {
            float3 p_film = make_float3(ss.p_film, 0);
            float3 p_sensor = _raster_to_camera.apply_point(p_film);
            auto c2w = camera_to_world();
            auto origin = c2w.apply_point(make_float3(0,0,0));
            auto direction = c2w.apply_vector(normalize(p_sensor));
            *ray = Ray(origin, direction);

            return 1;
        }

        PinholeCamera PinholeCamera::create(const SensorConfig &config) {
            auto transform = config.transform_config.create();
            return PinholeCamera(transform.mat4x4(), config.fov_y, config.velocity);
        }
    }
}