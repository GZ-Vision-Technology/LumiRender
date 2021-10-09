//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "camera_base.h"


namespace luminous {
    inline namespace render {

        class PinholeCamera : BASE_CLASS(CameraBase) {
        public:
            REFL_CLASS(PinholeCamera)

            CPU_ONLY(explicit PinholeCamera(const SensorConfig &config)
                             : PinholeCamera(config.transform_config.create().mat4x4(),
                             config.fov_y,
                             config.velocity) {})

            PinholeCamera(float4x4 m, float fov_y, float velocity);

            LM_XPU float generate_ray(const SensorSample &ss, Ray *ray);

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s:%s", type_name(this), CameraBase::to_string().c_str());
                            })
        };

    } // luminous::render
} // luminous