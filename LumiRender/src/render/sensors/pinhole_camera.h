//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "../include/sensor.h"

namespace luminous {
    inline namespace render {

        class PinholeCamera : public CameraBase {
        public:
            GEN_CLASS_NAME(PinholeCamera)

            NDSC std::string to_string() const;

            static PinholeCamera* create(const SensorConfig &config);
        };

    } // luminous::render
} // luminous