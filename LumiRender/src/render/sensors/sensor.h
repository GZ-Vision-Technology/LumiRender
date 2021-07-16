//
// Created by Zero on 2021/3/20.
//


#pragma once


#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "../include/config.h"
#include "pinhole_camera.h"
#include "thin_lens_camera.h"



namespace luminous {
    inline namespace render {
        using lstd::Variant;

        class Sensor : public Variant<PinholeCamera, ThinLensCamera> {
        public:
            using Variant::Variant;

            GEN_BASE_NAME(Sensor)

            NDSC_XPU float3 position() const;

            XPU void set_position(float3 pos);

            XPU float generate_ray(const SensorSample &ss, Ray * ray);

            XPU void set_film(const Film &film);

            XPU void update_film_resolution(uint2 res);

            NDSC_XPU Film *film();

            NDSC_XPU uint2 resolution() const;

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU Transform camera_to_world_rotation() const;

            NDSC_XPU float3 forward() const;

            NDSC_XPU float3 up() const;

            NDSC_XPU float3 right() const;

            XPU void move(float3 delta);

            NDSC_XPU float yaw() const;

            XPU void set_yaw(float yaw);

            XPU void update_yaw(float val);

            NDSC_XPU float pitch() const;

            XPU void set_pitch(float val);

            XPU void update_pitch(float val);

            NDSC_XPU float fov_y() const;

            XPU void set_fov_y(float val);

            XPU void update_fov_y(float val);

            NDSC_XPU float velocity() const;

            XPU void set_velocity(float val);

            NDSC_XPU float sensitivity() const;

            XPU void set_sensitivity(float val);

            GEN_TO_STRING_FUNC

            CPU_ONLY(static Sensor create(const SensorConfig &config);)
        };
    }
}