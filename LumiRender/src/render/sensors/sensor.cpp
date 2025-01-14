//
// Created by Zero on 2021/3/4.
//

#include "common.h"
#include "sensor.h"
#include "core/refl/factory.h"


namespace luminous {
    inline namespace render {


        void Sensor::update_param(float4x4 m, float fov_y) {
            LUMINOUS_VAR_PTR_DISPATCH(update_param, m, fov_y);
        }

        float Sensor::lens_radius() const {
            LUMINOUS_VAR_PTR_DISPATCH(lens_radius);
        }

        void Sensor::set_lens_radius(float r) {
            LUMINOUS_VAR_PTR_DISPATCH(set_lens_radius, r);
        }

        void Sensor::update_lens_radius(float d) {
            set_lens_radius(lens_radius() + d);
        }

        float Sensor::focal_distance() const {
            LUMINOUS_VAR_PTR_DISPATCH(focal_distance);
        }

        void Sensor::set_focal_distance(float fd) {
            LUMINOUS_VAR_PTR_DISPATCH(set_focal_distance, fd);
        }

        void Sensor::update_focal_distance(float d) {
            set_focal_distance(focal_distance() + d);
        }

        float3 Sensor::position() const {
            LUMINOUS_VAR_PTR_DISPATCH(position);
        }

        void Sensor::set_filter(const Filter &filter) {
            LUMINOUS_VAR_PTR_DISPATCH(set_filter, filter);
        }

        const Filter *Sensor::filter() const {
            LUMINOUS_VAR_PTR_DISPATCH(filter);
        }

        void Sensor::set_film(Film *film) {
            LUMINOUS_VAR_PTR_DISPATCH(set_film, film);
        }

        void Sensor::update_film_resolution(uint2 res) {
            LUMINOUS_VAR_PTR_DISPATCH(update_film_resolution, res);
        }

        Film *Sensor::film() {
            LUMINOUS_VAR_PTR_DISPATCH(film);
        }

        uint2 Sensor::resolution() const {
            LUMINOUS_VAR_PTR_DISPATCH(resolution);
        }

        void Sensor::set_fov_y(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_fov_y, val);
        }

        void Sensor::set_pitch(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_pitch, val);
        }

        void Sensor::set_yaw(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_yaw, val);
        }

        float Sensor::yaw() const {
            LUMINOUS_VAR_PTR_DISPATCH(yaw);
        }

        void Sensor::update_yaw(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_yaw, val);
        }

        float Sensor::pitch() const {
            LUMINOUS_VAR_PTR_DISPATCH(pitch);
        }

        void Sensor::update_pitch(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_pitch, val);
        }

        float Sensor::fov_y() const {
            LUMINOUS_VAR_PTR_DISPATCH(fov_y);
        }

        void Sensor::update_fov_y(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(update_fov_y, val);
        }

        float Sensor::velocity() const {
            LUMINOUS_VAR_PTR_DISPATCH(velocity);
        }

        void Sensor::set_velocity(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_velocity, val);
        }

        float Sensor::sensitivity() const {
            LUMINOUS_VAR_PTR_DISPATCH(sensitivity);
        }

        void Sensor::set_sensitivity(float val) {
            LUMINOUS_VAR_PTR_DISPATCH(set_sensitivity, val);
        }

        void Sensor::set_position(float3 pos) {
            LUMINOUS_VAR_PTR_DISPATCH(set_position, pos);
        }

        void Sensor::move(float3 delta) {
            LUMINOUS_VAR_PTR_DISPATCH(move, delta);
        }

        Transform Sensor::camera_to_world() const {
            LUMINOUS_VAR_PTR_DISPATCH(camera_to_world)
        }

        Transform Sensor::camera_to_world_rotation() const {
            LUMINOUS_VAR_PTR_DISPATCH(camera_to_world_rotation)
        }

        std::pair<float, Ray> Sensor::generate_ray(const SensorSample &ss) {
            Ray ray{};
            float weight = this->dispatch([&](auto &&self) -> decltype(auto) { return self->generate_ray(ss, &ray); });
            return {weight, ray};
        }

        float Sensor::generate_ray(const SensorSample &ss, Ray *ray) {
            LUMINOUS_VAR_PTR_DISPATCH(generate_ray, ss, ray);
        }

        float3 Sensor::forward() const {
            LUMINOUS_VAR_PTR_DISPATCH(forward)
        }

        float3 Sensor::up() const {
            LUMINOUS_VAR_PTR_DISPATCH(up)
        }

        float3 Sensor::right() const {
            LUMINOUS_VAR_PTR_DISPATCH(right)
        }

        CPU_ONLY(std::string Sensor::to_string() const {
            LUMINOUS_VAR_PTR_DISPATCH(to_string);
        })

        CPU_ONLY(std::pair<Sensor, std::vector<size_t>> Sensor::create(const SensorConfig &config) {
            auto ret = detail::create_ptr<Sensor>(config);
            auto film = Creator<Film>::create_ptr(config.film_config).first;
            ret.first.set_film(film);
            auto filter = Filter::create(config.filter_config);
            ret.first.set_filter(filter);
            return ret;
        })

    }
}