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
            float3 _position;
            float _fovy;
            float _yaw{};
            float _pitch{};
            float _velocity{};
        public:
            XPU CameraBase(float3 pos = make_float3(0), float fovy = 0);

            NDSC_XPU float3 position() const;

            NDSC_XPU float yaw() const;

            NDSC_XPU float pitch() const;

            NDSC_XPU float fovy() const;

            NDSC_XPU float velocity() const;
        };


        using lstd::Variant;
        class SensorHandle : public Variant<PinholeCamera*, PerspectiveCamera*> {
        public:
            using Variant::Variant;

            explicit SensorHandle(void *) {}
        };


    }
}
