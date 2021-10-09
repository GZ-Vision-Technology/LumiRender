//
// Created by Zero on 2021/1/15.
//


#pragma once

#include "base_libs/optics/common.h"
#include "base_libs/geometry/common.h"
#include "../films/film.h"
#include "../samplers/sampler_base.h"
#include "render/include/creator.h"
#include "core/concepts.h"
#include "core/refl/reflection.h"

namespace luminous {
    inline namespace render {
        class CameraBase : BASE_CLASS() {
        public:
            REFL_CLASS(CameraBase)
        protected:
            constexpr static float z_near = 0.01f;
            constexpr static float z_far = 1000.f;
            constexpr static float fov_max = 120.f;
            constexpr static float fov_min = 20.f;
            constexpr static float pitch_max = 80.f;
            constexpr static float3 right_vec = make_float3(1, 0, 0);
            constexpr static float3 up_vec = make_float3(0, 1, 0);
            constexpr static float3 forward_vec = make_float3(0, 0, 1);
            float3 _position;
            float _fov_y{0};
            float _yaw{};
            float _pitch{};
            float _velocity{};
            float _sensitivity{1.f};
            Transform _raster_to_screen{};
            Transform _camera_to_screen{};
            Transform _raster_to_camera{};
            Film _film;
            LM_XPU void _update(const float4x4 &m);

            LM_XPU void _update_raster();

            LM_XPU void _set_resolution(uint2 res);

        public:
            LM_XPU explicit CameraBase(float3 pos = make_float3(0), float fov_y = 30);

            LM_XPU CameraBase(const float4x4 m, float fov_y, float velocity);

            LM_XPU void update_film_resolution(uint2 res);

            LM_XPU void set_film(const Film &film);

            LM_ND_XPU Film *film();

            LM_ND_XPU uint2 resolution() const;

            LM_ND_XPU Transform camera_to_world() const;

            LM_ND_XPU Transform camera_to_world_rotation() const;

            LM_ND_XPU float3 forward() const;

            LM_ND_XPU float3 up() const;

            LM_ND_XPU float3 right() const;

            LM_ND_XPU float3 position() const;

            LM_XPU void set_position(float3 pos);

            LM_XPU void move(float3 delta);

            LM_ND_XPU float yaw() const;

            LM_XPU void set_yaw(float yaw);

            LM_XPU void update_yaw(float val);

            LM_ND_XPU float pitch() const;

            LM_XPU void set_pitch(float pitch);

            LM_XPU void update_pitch(float val);

            LM_ND_XPU float fov_y() const;

            LM_XPU void set_fov_y(float val);

            LM_XPU void update_fov_y(float val);

            LM_ND_XPU float velocity() const;

            LM_XPU void set_velocity(float val);

            LM_ND_XPU float sensitivity() const;

            LM_XPU void set_sensitivity(float val);

            GEN_STRING_FUNC({
                return string_printf("{fov_y:%f, position:%s, yaw:%f, pitch:%f, velocity : %f}",
                                     fov_y(),
                                     position().to_string().c_str(),
                                     yaw(),
                                     pitch(),
                                     velocity());
            })
        }; // luminous::render::CameraBase
    } // luminous::render
} // luminous
