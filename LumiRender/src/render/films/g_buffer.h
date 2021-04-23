//
// Created by Zero on 2021/3/5.
//


#pragma once

#include "film_base.h"

namespace luminous {
    inline namespace render {


        class GBufferFilm : public FilmBase {
        public:
            GBufferFilm(uint2 res) : FilmBase(res) {}

            NDSC std::string to_string() const;

            static GBufferFilm create(const FilmConfig &config);
        };
    }
}