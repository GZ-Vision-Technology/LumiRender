//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "perspective_camera.h"

namespace luminous {
    inline namespace render {
        PerspectiveCamera::PerspectiveCamera(const float4x4 m, float fov_y, float velocity)
                : CameraBase(m, fov_y, velocity) {}

        std::string PerspectiveCamera::to_string() const {
            return string_printf("%s: {fov_y:%f}", name(), fov_y());
        }

        PerspectiveCamera *PerspectiveCamera::create(const SensorConfig &config) {
            auto transform = config.transform_config.create();
            return new PerspectiveCamera(transform.mat4x4(), config.fov_y, config.velocity);
        }
    }
}