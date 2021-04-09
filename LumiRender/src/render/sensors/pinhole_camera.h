//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "camera_base.h"


namespace luminous {
    inline namespace render {

        class PinholeCamera : public CameraBase {
        public:
            GEN_CLASS_NAME(PinholeCamera)

            PinholeCamera(const float4x4 m, float fov_y, float velocity);

            XPU float generate_ray(const SensorSample &ss, Ray * ray);

            NDSC std::string to_string() const;

            static PinholeCamera create(const SensorConfig &config);
        };

    } // luminous::render
} // luminous