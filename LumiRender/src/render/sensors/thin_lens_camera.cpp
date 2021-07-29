//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "thin_lens_camera.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {
        ThinLensCamera::ThinLensCamera(const float4x4 m, float fov_y, float velocity)
                : CameraBase(m, fov_y, velocity) {}

        float ThinLensCamera::generate_ray(const SensorSample &ss, Ray *ray) {
            float3 p_film = make_float3(ss.p_film, 0);
            float3 p_sensor = _raster_to_camera.apply_point(p_film);
            auto c2w = camera_to_world();

            auto origin = c2w.apply_point(make_float3(0,0,0));
            auto direction = c2w.apply_vector(normalize(p_sensor));
            *ray = Ray(origin, direction);

            return 1;
        }

        CPU_ONLY(ThinLensCamera ThinLensCamera::create(const SensorConfig &config) {
            auto transform = config.transform_config.create();
            return ThinLensCamera(transform.mat4x4(), config.fov_y, config.velocity);
        })
    }
}