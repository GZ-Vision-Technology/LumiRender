//
// Created by Zero on 2021/1/15.
//


#pragma once

#include "graphics/optics/common.h"
#include "graphics/geometry/common.h"
#include "core/concepts.h"
#include "graphics/lstd/lstd.h"
#include "scene_graph.h"

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
            float _fov_y;
            float _yaw{};
            float _pitch{};
            float _velocity{};

            void _update(const float4x4 &m);

        public:
            XPU CameraBase(float3 pos = make_float3(0), float fov_y = 30);

            XPU CameraBase(const float4x4 m, float fov_y, float velocity);

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU Transform camera_to_world_rotation() const;

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

            NDSC_XPU float fov_y() const;

            XPU void update_fov_y(float val);

            NDSC_XPU float velocity() const;

            XPU void set_velocity(float val);

            NDSC std::string _to_string() const {
                return string_printf("{fov_y:%f, position:%s, yaw:%f, pitch:%f}",
                                     fov_y(),
                                     position().to_string().c_str(),
                                     yaw(),
                                     pitch());
            }
        };


        using lstd::Variant;

        class SensorHandle : public Variant<PinholeCamera *, PerspectiveCamera *> {
            using Variant::Variant;
        public:
            NDSC_XPU float3 position() const;

            XPU void set_position(float3 pos);

            NDSC_XPU Transform camera_to_world() const;

            NDSC_XPU const char *name();

            NDSC_XPU Transform camera_to_world_rotation() const;

            NDSC_XPU float3 forward() const;

            NDSC_XPU float3 up() const;

            NDSC_XPU float3 right() const;

            XPU void move(float3 delta);

            NDSC_XPU float yaw() const;

            XPU void update_yaw(float val);

            NDSC_XPU float pitch() const;

            XPU void update_pitch(float val);

            NDSC_XPU float fov_y() const;

            XPU void update_fov_y(float val);

            NDSC_XPU float velocity() const;

            XPU void set_velocity(float val);

            NDSC std::string to_string() const;

            static SensorHandle create(const SensorConfig &config);
        };
    }
}
