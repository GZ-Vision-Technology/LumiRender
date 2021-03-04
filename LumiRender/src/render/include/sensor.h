//
// Created by Zero on 2021/1/15.
//


#pragma once

#include "graphics/optics/common.h"
#include "graphics/geometry/common.h"
#include "core/concepts.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {

        class PinholeCamera;

        class PerspectiveCamera;

        class CameraBase {
        protected:
            constexpr static float fov_max = 120.f;
            constexpr static float fov_min = 20.f;
            constexpr static float pitch_max = 80.f;
            constexpr static float3 right_vec = make_float3(1, 0, 0);
            constexpr static float3 up_vec = make_float3(0, 1, 0);
            constexpr static float3 forward_vec = make_float3(0, 0, 1);
            float3 _position;
            float _fovy;
            float _yaw{};
            float _pitch{};
            float _velocity{};
        public:
            XPU CameraBase(float3 pos = make_float3(0), float fovy = 30);

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU Transform linear_space() const;

            NDSC_XPU float3 forward() const;

            NDSC_XPU float3 up() const;

            NDSC_XPU float3 right() const;

            NDSC_XPU float3 position() const;

            XPU void set_position(float3 pos);

            XPU void move(float3 delta);

            NDSC_XPU float yaw() const;

            XPU void update_yaw(float val);

            NDSC_XPU float pitch() const;

            XPU void update_pitch(float val);

            NDSC_XPU float fovy() const;

            XPU void update_fovy(float val);

            NDSC_XPU float velocity() const;

            XPU void set_velocity(float val);
        };


        using lstd::Variant;

        class SensorHandle : public Variant<PinholeCamera *, PerspectiveCamera *> {
            using Variant::Variant;
        public:
            NDSC_XPU float3 position() const;

            XPU void set_position(float3 pos);

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU Transform linear_space() const;

            NDSC_XPU float3 forward() const;

            NDSC_XPU float3 up() const;

            NDSC_XPU float3 right() const;

            XPU void move(float3 delta);

            NDSC_XPU float yaw() const;

            XPU void update_yaw(float val);

            NDSC_XPU float pitch() const;

            XPU void update_pitch(float val);

            NDSC_XPU float fovy() const;

            XPU void update_fovy(float val);

            NDSC_XPU float velocity() const;

            XPU void set_velocity(float val);
        };
    }
}
