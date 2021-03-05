//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "perspective_camera.h"

namespace luminous {
    inline namespace render {

        std::string PerspectiveCamera::to_string() const {
            return string_printf("%s: {fovy:%f}", name(), fovy());
        }

        PerspectiveCamera *PerspectiveCamera::create(const SensorConfig &config) {
            return nullptr;
        }
    }
}