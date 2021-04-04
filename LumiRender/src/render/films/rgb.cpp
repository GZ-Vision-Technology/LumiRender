//
// Created by Zero on 2021/3/5.
//

#include "rgb.h"
#include "graphics/optics/rgb.h"

namespace luminous {
    inline namespace render {

        void RGBFilm::add_sample(float2 p_film, float3 color, float weight) {
            auto p = make_int2(p_film);
            _d_frame_buffer[p.y * _resolution.x + p.x] = make_rgba(color * weight);
        }

        std::string RGBFilm::to_string() const {
            return string_printf("%s : {resolution :%s}", name(),
                                 _resolution.to_string().c_str());
        }

        RGBFilm RGBFilm::create(const FilmConfig &config) {
            return RGBFilm(config.resolution);
        }
    } // luminous::render
} // luminous