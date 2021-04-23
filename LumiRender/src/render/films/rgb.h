//
// Created by Zero on 2021/3/5.
//


#pragma once


#include "film_base.h"

namespace luminous {
    inline namespace render {

        class RGBFilm : public FilmBase {
        public:
            RGBFilm(uint2 res) : FilmBase(res) {}

            XPU void add_sample(uint2 pixel, float3 color, float weight, uint frame_index = 0u);

            NDSC std::string to_string() const;

            static RGBFilm create(const FilmConfig &config);
        };
    }
}