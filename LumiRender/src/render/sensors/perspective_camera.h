//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "sensor.h"

namespace luminous {

    inline namespace render {
        class PerspectiveCamera : public CameraBase {
        private:
            float _lens_radius{0};

            // distance of focal plane to center of lens
            float _focal_distance;
        public:
            PerspectiveCamera(const float4x4 m, float fov_y, float velocity);

            GEN_CLASS_NAME(PerspectiveCamera)

            XPU float generate_ray(const SensorSample &ss, Ray *ray);

            NDSC std::string to_string() const;

            static PerspectiveCamera create(const SensorConfig &config);
        };
    }

} // luminous