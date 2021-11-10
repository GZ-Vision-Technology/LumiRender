//
// Created by Zero on 2021/3/20.
//


#pragma once


#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "../include/config.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;

        class PinholeCamera;

        class ThinLensCamera;

        struct SensorSample;

        class Film;

        class Sensor : BASE_CLASS(Variant < ThinLensCamera * >)

    {
        public:
        REFL_CLASS(Sensor)

        using BaseBinder::BaseBinder;

        GEN_BASE_NAME(Sensor)

        LM_ND_XPU float3 position() const;

        LM_XPU void set_position(float3 pos);

        LM_ND_XPU float lens_radius() const;

        LM_XPU void set_lens_radius(float r);

        LM_XPU void update_lens_radius(float d);

        LM_ND_XPU float focal_distance() const;

        LM_XPU void set_focal_distance(float fd);

        LM_XPU void update_focal_distance(float d);

        LM_ND_XPU float generate_ray(const SensorSample &ss, Ray *ray);

        LM_ND_XPU std::pair<float, Ray> generate_ray(const SensorSample &ss);

        LM_XPU void set_film(Film *film);

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

        CPU_ONLY(static std::pair<Sensor, std::vector<size_t>> create(const SensorConfig &config);)
    };
}
}