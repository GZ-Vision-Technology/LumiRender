//
// Created by Zero on 2021/3/4.
//

#include <render/include/sensor.h>

#include "pinhole_camera.h"
#include "geometry_surface.h"
#include "perspective_camera.h"

namespace luminous {
    inline namespace render {
        CameraBase::CameraBase(float3 pos, float fov_y)
                : _position(pos), _fov_y(fov_y) {}

        CameraBase::CameraBase(const float4x4 m, float fov_y, float velocity)
                : _fov_y(fov_y),
                  _velocity(velocity) {
            _update(m);
            _camera_to_screen = Transform::perspective(fov_y, 0.01, 1000);
        }

        void CameraBase::_update(const float4x4 &m) {
            float sy = sqrt(sqr(m[2][1]) + sqr(m[2][2]));
            _pitch = degrees(-std::atan2(m[2][1], m[2][2]));
            _yaw = degrees(-std::atan2(-m[2][0], sy));
            _position = make_float3(m[3]);
            cout << _position.to_string();
        }

        void CameraBase::set_film(const FilmHandle &film) {
            _film = film;
            float2 res = _film.resolution();
            box2f scrn = _film.screen_window();
            float2 span = scrn.span();
            Transform screen_to_raster = Transform::scale(res.x, res.y, 1) *
                                         Transform::scale(1 / span.x, 1 / -span.y, 1) *
                                         Transform::translate(make_float3(-scrn.lower.x, -scrn.upper.y, 0));
            _raster_to_screen = screen_to_raster.inverse();
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

        void CameraBase::update_fov_y(float val) {
            float new_fovy = _fov_y + val;
            if (new_fovy > fov_max) {
                _fov_y = fov_max;
            } else if (new_fovy < fov_min) {
                _fov_y = fov_min;
            } else {
                _fov_y += val;
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
            return translation * camera_to_world_rotation();
        }

        Transform CameraBase::camera_to_world_rotation() const {
            auto horizontal = Transform::rotation_y(_yaw);
            auto vertical = Transform::rotation_x(_pitch);
            return vertical * horizontal;
        }

        float3 CameraBase::forward() const {
            return camera_to_world_rotation().inverse().apply_vector(forward_vec);
        }

        float3 CameraBase::up() const {
            return camera_to_world_rotation().inverse().apply_vector(up_vec);
        }

        float3 CameraBase::right() const {
            return camera_to_world_rotation().inverse().apply_vector(right_vec);
        }

        float3 SensorHandle::position() const {
            LUMINOUS_VAR_PTR_DISPATCH(position);
        }

        void SensorHandle::set_film(const Film &film) {
            LUMINOUS_VAR_PTR_DISPATCH(set_film, film);
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

        float SensorHandle::fov_y() const {
            LUMINOUS_VAR_PTR_DISPATCH(fov_y);
        }

        void SensorHandle::update_fov_y(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_fov_y, val);
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

        Transform SensorHandle::camera_to_world_rotation() const {
            LUMINOUS_VAR_PTR_DISPATCH(camera_to_world_rotation)
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

        const char *SensorHandle::name() {
            LUMINOUS_VAR_PTR_DISPATCH(name)
        }

        std::string SensorHandle::to_string() const {
            LUMINOUS_VAR_PTR_DISPATCH(to_string)
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC SensorHandle create_sensor(const SensorConfig &config) {
                using Sensor = std::remove_pointer_t<std::tuple_element_t<current_index, SensorHandle::TypeTuple>>;
                if (Sensor::name() == config.type) {
                    return SensorHandle(Sensor::create(config));
                }
                return create_sensor<current_index + 1>(config);
            }

            template<>
            NDSC SensorHandle create_sensor<std::tuple_size_v<SensorHandle::TypeTuple>>(const SensorConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        SensorHandle SensorHandle::create(const SensorConfig &config) {
            return detail::create_sensor<0>(config);
        }
    }
}