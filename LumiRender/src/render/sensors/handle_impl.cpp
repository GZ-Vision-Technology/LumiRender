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

        void CameraBase::update_yaw(float val) {
            _yaw += val;
            if (_yaw > 360) {
                _yaw = fmodf(_yaw, 360);
            } else if (_yaw < 0) {
                _yaw = 360 - fmodf(_yaw, 360);
            }
        }

        void CameraBase::update_pitch(float val) {
            float p = _pitch + val;
            if (p > pitch_max) {
                _pitch = pitch_max;
            } else if (p < -pitch_max) {
                _pitch = -pitch_max;
            } else {
                _pitch = p;
            }
        }

        void CameraBase::update_fovy(float val) {
            float new_fovy = _fovy + val;
            if (new_fovy > fov_max) {
                _fovy = fov_max;
            } else if (new_fovy < fov_min) {
                _fovy = fov_min;
            } else {
                _fovy += val;
            }
        }

        void CameraBase::set_velocity(float val) {
            _velocity = val;
        }

        void CameraBase::set_position(float3 pos) {
            _position = pos;
        }

        void CameraBase::move(float3 delta) {
            _position += delta;
        }

        Transform CameraBase::camera_to_world() const {
            auto translation = Transform::translation(_position);
            return translation * linear_space();
        }

        Transform CameraBase::linear_space() const {
            auto horizontal = Transform::rotation_y(_yaw);
            auto vertical = Transform::rotation_x(_pitch);
            return vertical * horizontal;
        }

        float3 CameraBase::forward() const {
            return linear_space().inverse().apply_vector(forward_vec);
        }

        float3 CameraBase::up() const {
            return linear_space().inverse().apply_vector(up_vec);
        }

        float3 CameraBase::right() const {
            return linear_space().inverse().apply_vector(right_vec);
        }

        float3 SensorHandle::position() const {
            LUMINOUS_VAR_PTR_DISPATCH(position);
        }

        float SensorHandle::yaw() const {
            LUMINOUS_VAR_PTR_DISPATCH(yaw);
        }

        void SensorHandle::update_yaw(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_yaw, val);
        }

        float SensorHandle::pitch() const {
            LUMINOUS_VAR_PTR_DISPATCH(pitch);
        }

        void SensorHandle::update_pitch(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_pitch, val);
        }

        float SensorHandle::fovy() const {
            LUMINOUS_VAR_PTR_DISPATCH(fovy);
        }

        void SensorHandle::update_fovy(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_fovy, val);
        }

        float SensorHandle::velocity() const {
            LUMINOUS_VAR_PTR_DISPATCH(velocity);
        }

        void SensorHandle::set_velocity(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_velocity, val);
        }

        void SensorHandle::set_position(float3 pos) {
            LUMINOUS_VAR_PTR_DISPATCH(set_position, pos);
        }

        void SensorHandle::move(float3 delta) {
            LUMINOUS_VAR_PTR_DISPATCH(move, delta);
        }

        Transform SensorHandle::camera_to_world() const {
            LUMINOUS_VAR_PTR_DISPATCH(camera_to_world)
        }

        Transform SensorHandle::linear_space() const {
            LUMINOUS_VAR_PTR_DISPATCH(linear_space)
        }

        float3 SensorHandle::forward() const {
            LUMINOUS_VAR_PTR_DISPATCH(forward)
        }

        float3 SensorHandle::up() const {
            LUMINOUS_VAR_PTR_DISPATCH(up)
        }

        float3 SensorHandle::right() const {
            LUMINOUS_VAR_PTR_DISPATCH(right)
        }
    }
}