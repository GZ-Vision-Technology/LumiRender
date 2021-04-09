//
// Created by Zero on 2021/3/5.
//

#include "film.h"

namespace luminous {
    inline namespace render {
        const char *Film::name() {
            LUMINOUS_VAR_DISPATCH(name);
        }

        uint2 Film::resolution() const {
            LUMINOUS_VAR_DISPATCH(resolution);
        }

        Box2f Film::screen_window() const {
            LUMINOUS_VAR_DISPATCH(screen_window);
        }

        void Film::set_resolution(uint2 res) {
            LUMINOUS_VAR_DISPATCH(set_resolution, res);
        }

        void Film::add_sample(float2 p_film, float3 color, float weight, uint frame_index) {
            LUMINOUS_VAR_DISPATCH(add_sample, p_film, color, weight, frame_index);
        }

        void Film::set_accumulate_buffer(float4 *d_ptr) {
            LUMINOUS_VAR_DISPATCH(set_accumulate_buffer, d_ptr);
        }

        void Film::set_frame_buffer(FrameBufferType *d_ptr) {
            LUMINOUS_VAR_DISPATCH(set_frame_buffer, d_ptr);
        }

        std::string Film::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string);
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC Film create_film(const FilmConfig &config) {
                using Class = std::remove_pointer_t<std::tuple_element_t<current_index, Film::TypeTuple>>;
                if (Class::name() == config.type) {
                    return Film(Class::create(config));
                }
                return create_film<current_index + 1>(config);
            }

            template<>
            NDSC Film create_film<std::tuple_size_v<Film::TypeTuple>>(const FilmConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        Film Film::create(const FilmConfig &config) {
            return detail::create_film<0>(config);
        }

    }
}