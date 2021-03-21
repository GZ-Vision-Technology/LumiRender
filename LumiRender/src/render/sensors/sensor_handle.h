//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "sensor.h"
#include "pinhole_camera.h"
#include "perspective_camera.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;

        class SensorHandle : public Variant<PinholeCamera, PerspectiveCamera> {
        public:
            using Variant::Variant;

            NDSC_XPU float3 position() const;

            XPU void set_position(float3 pos);

            XPU float generate_ray(const SensorSample &ss, Ray * ray);

            XPU void set_film(const FilmHandle &film);

            NDSC_XPU FilmHandle *film();

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU const char *name();

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

            NDSC std::string to_string() const;

            static SensorHandle create(const SensorConfig &config);
        };
    }
}