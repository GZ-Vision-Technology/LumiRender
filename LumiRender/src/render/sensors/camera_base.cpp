//
// Created by Zero on 2021/3/20.
//

#include "camera_base.h"
#include "film.h"

namespace luminous {
    inline namespace render {

        CameraBase::CameraBase(float3 pos, float fov_y)
                : _position(pos), _fov_y(fov_y) {}

        CameraBase::CameraBase(float4x4 m, float fov_y, float velocity)
                : _velocity(velocity) {
            _update(m);
            set_fov_y(fov_y);
        }

        void CameraBase::_update_raster() {
            _camera_to_screen = Transform::perspective(_fov_y, z_near, z_far);
            _raster_to_camera = _camera_to_screen.inverse() * _raster_to_screen;
        }

        void CameraBase::_update(const float4x4 &m) {
            float sy = sqrt(sqr(m[2][1]) + sqr(m[2][2]));
            _pitch = degrees(-std::atan2(m[2][1], m[2][2]));
            _yaw = degrees(-std::atan2(-m[2][0], sy));
            _position = make_float3(m[3]);
        }

        void CameraBase::update_film_resolution(uint2 res) {
            _film->set_resolution(res);
            _set_resolution(res);
        }

        void CameraBase::_set_resolution(uint2 res) {
            Box2f scrn = _film->screen_window();
            float2 span = scrn.span();
            Transform screen_to_raster = Transform::scale(res.x, res.y, 1) *
                                         Transform::scale(1 / span.x, 1 / -span.y, 1) *
                                         Transform::translation(-scrn.lower.x, -scrn.upper.y, 0);
            _raster_to_screen = screen_to_raster.inverse();
            _update_raster();
        }

        void CameraBase::set_film(Film *film) {
            _film = film;
            _set_resolution(_film->resolution());
            _update_raster();
        }

        uint2 CameraBase::resolution() const {
            return _film->resolution();
        }

        Film *CameraBase::film() {
            return _film;
        }

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

        float CameraBase::fov_y() const {
            return _fov_y;
        }

        void CameraBase::set_yaw(float yaw) {
            _yaw = yaw;
            if (_yaw > 360) {
                _yaw = fmodf(_yaw, 360);
            } else if (_yaw < 0) {
                _yaw = 360 + fmodf(_yaw, 360);
            }
        }

        void CameraBase::set_pitch(float pitch) {
            if (pitch > pitch_max) {
                _pitch = pitch_max;
            } else if (pitch < -pitch_max) {
                _pitch = -pitch_max;
            } else {
                _pitch = pitch;
            }
        }

        void CameraBase::update_yaw(float val) {
            set_yaw(_yaw + val);
        }

        void CameraBase::update_pitch(float val) {
            set_pitch(_pitch + val);
        }

        void CameraBase::set_fov_y(float new_fov_y) {
            if (new_fov_y > fov_max) {
                _fov_y = fov_max;
            } else if (new_fov_y < fov_min) {
                _fov_y = fov_min;
            } else {
                _fov_y = new_fov_y;
            }
            _update_raster();
        }

        void CameraBase::update_fov_y(float val) {
            set_fov_y(_fov_y + val);
        }

        void CameraBase::set_velocity(float val) {
            _velocity = val;
        }

        void CameraBase::set_position(float3 pos) {
            _position = pos;
        }

        float CameraBase::sensitivity() const {
            return _sensitivity;
        }

        void CameraBase::set_sensitivity(float val) {
            _sensitivity = val;
        }

        void CameraBase::move(float3 delta) {
            _position += delta;
        }

        Transform CameraBase::camera_to_world() const {
            auto translation = Transform::translation(_position);
            return translation * camera_to_world_rotation();
        }

        Transform CameraBase::camera_to_world_rotation() const {
            auto horizontal = Transform::rotation_y(_yaw);
            auto vertical = Transform::rotation_x(-_pitch);
            return horizontal *vertical;
        }

        float3 CameraBase::forward() const {
            return camera_to_world_rotation().apply_vector(forward_vec);
        }

        float3 CameraBase::up() const {
            return camera_to_world_rotation().apply_vector(up_vec);
        }

        float3 CameraBase::right() const {
            return camera_to_world_rotation().apply_vector(right_vec);
        }

    } // luminous::render
} // luminous