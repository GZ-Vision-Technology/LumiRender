//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "pinhole_camera.h"

namespace luminous {
    inline namespace render {

        std::string PinholeCamera::to_string() const {
            return string_printf("%s: {fovy:%f}", name(), fovy());
        }

        PinholeCamera *PinholeCamera::create(const SensorConfig &config) {
            return nullptr;
        }
    }
}