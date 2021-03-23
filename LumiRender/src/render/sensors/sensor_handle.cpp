//
// Created by Zero on 2021/3/4.
//

#include "sensor_handle.h"

namespace luminous {
    inline namespace render {


        float3 SensorHandle::position() const {
            LUMINOUS_VAR_DISPATCH(position);
        }

        void SensorHandle::set_film(const FilmHandle &film) {
            LUMINOUS_VAR_DISPATCH(set_film, film);
        }

        FilmHandle *SensorHandle::film() {
            LUMINOUS_VAR_DISPATCH(film);
        }

        int2 SensorHandle::resolution() const {
            LUMINOUS_VAR_DISPATCH(resolution);
        }

        void SensorHandle::set_fov_y(float val) {
            LUMINOUS_VAR_DISPATCH(set_fov_y, val);
        }

        void SensorHandle::set_pitch(float val) {
            LUMINOUS_VAR_DISPATCH(set_pitch, val);
        }

        void SensorHandle::set_yaw(float val) {
            LUMINOUS_VAR_DISPATCH(set_yaw, val);
        }

        float SensorHandle::yaw() const {
            LUMINOUS_VAR_DISPATCH(yaw);
        }

        void SensorHandle::update_yaw(float val) {
            LUMINOUS_VAR_DISPATCH(update_yaw, val);
        }

        float SensorHandle::pitch() const {
            LUMINOUS_VAR_DISPATCH(pitch);
        }

        void SensorHandle::update_pitch(float val) {
            LUMINOUS_VAR_DISPATCH(update_pitch, val);
        }

        float SensorHandle::fov_y() const {
            LUMINOUS_VAR_DISPATCH(fov_y);
        }

        void SensorHandle::update_fov_y(float val) {
            LUMINOUS_VAR_DISPATCH(update_fov_y, val);
        }

        float SensorHandle::velocity() const {
            LUMINOUS_VAR_DISPATCH(velocity);
        }

        void SensorHandle::set_velocity(float val) {
            LUMINOUS_VAR_DISPATCH(set_velocity, val);
        }

        void SensorHandle::set_position(float3 pos) {
            LUMINOUS_VAR_DISPATCH(set_position, pos);
        }

        void SensorHandle::move(float3 delta) {
            LUMINOUS_VAR_DISPATCH(move, delta);
        }

        Transform SensorHandle::camera_to_world() const {
            LUMINOUS_VAR_DISPATCH(camera_to_world)
        }

        Transform SensorHandle::camera_to_world_rotation() const {
            LUMINOUS_VAR_DISPATCH(camera_to_world_rotation)
        }

        float SensorHandle::generate_ray(const SensorSample &ss, Ray *ray) {
            LUMINOUS_VAR_DISPATCH(generate_ray, ss, ray);
        }

        float3 SensorHandle::forward() const {
            LUMINOUS_VAR_DISPATCH(forward)
        }

        float3 SensorHandle::up() const {
            LUMINOUS_VAR_DISPATCH(up)
        }

        float3 SensorHandle::right() const {
            LUMINOUS_VAR_DISPATCH(right)
        }

        const char *SensorHandle::name() {
            LUMINOUS_VAR_DISPATCH(name)
        }

        std::string SensorHandle::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string)
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
            auto ret = detail::create_sensor<0>(config);
            ret.set_film(FilmHandle::create(config.film_config));
            return ret;
        }
    }
}