//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "pinhole_camera.h"

namespace luminous {
    inline namespace render {

        PinholeCamera::PinholeCamera(const float4x4 m, float fov_y, float velocity)
                : CameraBase(m, fov_y, velocity) {}

        std::string PinholeCamera::to_string() const {
            return string_printf("%s: {fov_y:%f}", name(), fov_y());
        }

        PinholeCamera *PinholeCamera::create(const SensorConfig &config) {
            auto transform = config.transform_config.create();
            return new PinholeCamera(transform.mat4x4(), config.fov_y, config.velocity);
        }


    }
}