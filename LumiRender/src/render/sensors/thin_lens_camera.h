//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "camera_base.h"
#include "render/include/config.h"

namespace luminous {

    inline namespace render {
        class ThinLensCamera : BASE_CLASS(CameraBase) {
        public:
            REFL_CLASS(ThinLensCamera)
        private:
            float _lens_radius{0};

            // distance of focal plane to center of lens
            float _focal_distance{};
        public:

            CPU_ONLY(explicit ThinLensCamera(const SensorConfig &config)
                    : ThinLensCamera(config.transform_config.create().mat4x4(),
                                     config.fov_y,
                                     config.velocity) {})

            ThinLensCamera(const float4x4 &m, float fov_y, float velocity);

            LM_XPU float generate_ray(const SensorSample &ss, Ray *ray);

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s:%s", type_name(this), CameraBase::to_string().c_str());
                            })
        };
    }

} // luminous