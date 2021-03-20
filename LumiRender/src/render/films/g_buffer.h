//
// Created by Zero on 2021/3/5.
//


#pragma once

#include "film.h"

namespace luminous {
    inline namespace render {


        class GBufferFilm : public FilmBase {
        public:
            GBufferFilm(int2 res) : FilmBase(res) {}

            GEN_CLASS_NAME(GBufferFilm)

            static GBufferFilm create(const FilmConfig &config);
        };
    }
}