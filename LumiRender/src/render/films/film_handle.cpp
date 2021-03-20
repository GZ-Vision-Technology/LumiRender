//
// Created by Zero on 2021/3/5.
//

#include "film_handle.h"

namespace luminous {
    inline namespace render {
        int2 FilmHandle::resolution() const {
            LUMINOUS_VAR_DISPATCH(resolution);
        }

        Box2f FilmHandle::screen_window() const {
            LUMINOUS_VAR_DISPATCH(screen_window);
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC FilmHandle create_film(const FilmConfig &config) {
                using Film = std::remove_pointer_t<std::tuple_element_t<current_index, FilmHandle::TypeTuple>>;
                if (Film::name() == config.type) {
                    return FilmHandle(Film::create(config));
                }
                return create_film<current_index + 1>(config);
            }

            template<>
            NDSC FilmHandle create_film<std::tuple_size_v<FilmHandle::TypeTuple>>(const FilmConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        FilmHandle FilmHandle::create(const FilmConfig &config) {
            return detail::create_film<0>(config);
        }
    }
}