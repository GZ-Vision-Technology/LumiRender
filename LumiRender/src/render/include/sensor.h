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
        private:
            float3 _position;
            float _fovy;
            float _yaw;
            float _pitch;
            float _velocity;
        public:
            CameraBase(float3 pos = make_float3(0), float _fovy = 0);

            float3 position() const;

            float yaw() const;

            float pitch() const;

            float fovy() const;

            float velocity() const;
        }

        using lstd::Variant;
        class SensorHandle : public Variant<PinholeCamera*, PerspectiveCamera*> {
        public:
            using Variant::Variant;

            explicit SensorHandle(void *) {}
        };


    }
}
