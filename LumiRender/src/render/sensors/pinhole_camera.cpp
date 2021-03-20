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
            return string_printf("%s:%s", name(), _to_string().c_str());
        }

        PinholeCamera PinholeCamera::create(const SensorConfig &config) {
            auto transform = config.transform_config.create();
            return PinholeCamera(transform.mat4x4(), config.fov_y, config.velocity);
        }
    }
}