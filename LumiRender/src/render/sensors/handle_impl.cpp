//
// Created by Zero on 2021/3/4.
//

#include <render/include/sensor.h>

#include "pinhole_camera.h"
#include "geometry_surface.h"
#include "perspective_camera.h"

namespace luminous {
    inline namespace render {
        CameraBase::CameraBase(float3 pos, float fovy)
                : _position(pos), _fovy(fovy) {}

        float3 CameraBase::position() const {
            return _position;
        }

        float CameraBase::yaw() const {
            return _yaw;
        }

        float CameraBase::pitch() const {
            return _pitch;
        }

        float CameraBase::velocity() const {
            return _velocity;
        }

        float CameraBase::fovy() const {
            return _fovy;
        }

    }
}