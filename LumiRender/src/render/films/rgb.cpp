//
// Created by Zero on 2021/3/5.
//

#include "rgb.h"

namespace luminous {
    inline namespace render {

        void RGBFilm::add_sample(float2 p_film, float3 color, float weight) {
            //todo
        }

        RGBFilm RGBFilm::create(const FilmConfig &config) {
            return RGBFilm(config.resolution);
        }
    } // luminous::render
} // luminous