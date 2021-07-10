//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "camera_base.h"

namespace luminous {

    inline namespace render {
        class PerspectiveCamera : public CameraBase {
        private:
            float _lens_radius{0};

            // distance of focal plane to center of lens
            float _focal_distance{};
        public:
            PerspectiveCamera(const float4x4 m, float fov_y, float velocity);

            XPU float generate_ray(const SensorSample &ss, Ray *ray);

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("%s:%s", type_name(this), CameraBase::to_string().c_str());
            })

            CPU_ONLY(static PerspectiveCamera create(const SensorConfig &config);)
        };
    }

} // luminous