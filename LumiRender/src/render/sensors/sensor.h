//
// Created by Zero on 2021/3/20.
//


#pragma once


#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "../include/config.h"
#include "pinhole_camera.h"
#include "thin_lens_camera.h"


namespace luminous {
    inline namespace render {
        using lstd::Variant;

        class Sensor : BASE_CLASS(Variant<PinholeCamera *, ThinLensCamera *>) {
        public:
            REFL_CLASS(Sensor)

            using BaseBinder::BaseBinder;

            GEN_BASE_NAME(Sensor)

            LM_ND_XPU float3 position() const;

            LM_XPU void set_position(float3 pos);

            LM_XPU float generate_ray(const SensorSample &ss, Ray *ray);

            LM_XPU void set_film(const Film &film);

            LM_XPU void update_film_resolution(uint2 res);

            LM_ND_XPU Film *film();

            LM_ND_XPU uint2 resolution() const;

            LM_ND_XPU Transform camera_to_world() const;

            LM_ND_XPU Transform camera_to_world_rotation() const;

            LM_ND_XPU float3 forward() const;

            LM_ND_XPU float3 up() const;

            LM_ND_XPU float3 right() const;

            LM_XPU void move(float3 delta);

            LM_ND_XPU float yaw() const;

            LM_XPU void set_yaw(float yaw);

            LM_XPU void update_yaw(float val);

            LM_ND_XPU float pitch() const;

            LM_XPU void set_pitch(float val);

            LM_XPU void update_pitch(float val);

            LM_ND_XPU float fov_y() const;

            LM_XPU void set_fov_y(float val);

            LM_XPU void update_fov_y(float val);

            LM_ND_XPU float velocity() const;

            LM_XPU void set_velocity(float val);

            LM_ND_XPU float sensitivity() const;

            LM_XPU void set_sensitivity(float val);

            CPU_ONLY(LM_NODISCARD std::string to_string() const;)

            CPU_ONLY(static Sensor create(const SensorConfig &config);)
        };
    }
}