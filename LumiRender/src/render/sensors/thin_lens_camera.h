//
// Created by Zero on 2021/3/4.
//


#pragma once

#include "camera_base.h"
#include "parser/config.h"

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
                                     config.velocity,
                                     config.lens_radius,
                                     config.focal_distance) {})

            ThinLensCamera(const float4x4 &m, float fov_y, float velocity,
                           float lens_radius, float focal_distance);

            LM_XPU float generate_ray(const SensorSample &ss, Ray *ray);

            LM_ND_XPU float lens_radius() const { return _lens_radius; }

            LM_XPU void set_lens_radius(float r) { r = r >= 0 ? r : 0; _lens_radius = r; };

            LM_ND_XPU float focal_distance() const { return _focal_distance; }

            LM_XPU void set_focal_distance(float fd) { fd = fd >= 0 ? fd : 0; _focal_distance = fd; }

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s:%s", type_name(this), CameraBase::to_string().c_str());
                            })
        };
    }

} // luminous