//
// Created by Zero on 2021/3/4.
//

#include "sensor.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {


        float3 Sensor::position() const {
            LUMINOUS_VAR_DISPATCH(position);
        }

        void Sensor::set_film(const Film &film) {
            LUMINOUS_VAR_DISPATCH(set_film, film);
        }

        void Sensor::update_film_resolution(uint2 res) {
            LUMINOUS_VAR_DISPATCH(update_film_resolution, res);
        }

        Film *Sensor::film() {
            LUMINOUS_VAR_DISPATCH(film);
        }

        uint2 Sensor::resolution() const {
            LUMINOUS_VAR_DISPATCH(resolution);
        }

        void Sensor::set_fov_y(float val) {
            LUMINOUS_VAR_DISPATCH(set_fov_y, val);
        }

        void Sensor::set_pitch(float val) {
            LUMINOUS_VAR_DISPATCH(set_pitch, val);
        }

        void Sensor::set_yaw(float val) {
            LUMINOUS_VAR_DISPATCH(set_yaw, val);
        }

        float Sensor::yaw() const {
            LUMINOUS_VAR_DISPATCH(yaw);
        }

        void Sensor::update_yaw(float val) {
            LUMINOUS_VAR_DISPATCH(update_yaw, val);
        }

        float Sensor::pitch() const {
            LUMINOUS_VAR_DISPATCH(pitch);
        }

        void Sensor::update_pitch(float val) {
            LUMINOUS_VAR_DISPATCH(update_pitch, val);
        }

        float Sensor::fov_y() const {
            LUMINOUS_VAR_DISPATCH(fov_y);
        }

        void Sensor::update_fov_y(float val) {
            LUMINOUS_VAR_DISPATCH(update_fov_y, val);
        }

        float Sensor::velocity() const {
            LUMINOUS_VAR_DISPATCH(velocity);
        }

        void Sensor::set_velocity(float val) {
            LUMINOUS_VAR_DISPATCH(set_velocity, val);
        }

        float Sensor::sensitivity() const {
            LUMINOUS_VAR_DISPATCH(sensitivity);
        }

        void Sensor::set_sensitivity(float val) {
            LUMINOUS_VAR_DISPATCH(set_sensitivity, val);
        }

        void Sensor::set_position(float3 pos) {
            LUMINOUS_VAR_DISPATCH(set_position, pos);
        }

        void Sensor::move(float3 delta) {
            LUMINOUS_VAR_DISPATCH(move, delta);
        }

        Transform Sensor::camera_to_world() const {
            LUMINOUS_VAR_DISPATCH(camera_to_world)
        }

        Transform Sensor::camera_to_world_rotation() const {
            LUMINOUS_VAR_DISPATCH(camera_to_world_rotation)
        }

        float Sensor::generate_ray(const SensorSample &ss, Ray *ray) {
            LUMINOUS_VAR_DISPATCH(generate_ray, ss, ray);
        }

        float3 Sensor::forward() const {
            LUMINOUS_VAR_DISPATCH(forward)
        }

        float3 Sensor::up() const {
            LUMINOUS_VAR_DISPATCH(up)
        }

        float3 Sensor::right() const {
            LUMINOUS_VAR_DISPATCH(right)
        }

        const char *Sensor::name() {
            LUMINOUS_VAR_DISPATCH(name)
        }

        std::string Sensor::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string)
        }

        Sensor Sensor::create(const SensorConfig &config) {
            auto ret = detail::create<Sensor>(config);
            ret.set_film(Film::create(config.film_config));
            return ret;
        }
    }
}