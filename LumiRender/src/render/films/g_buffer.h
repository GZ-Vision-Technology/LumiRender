//
// Created by Zero on 2021/3/5.
//


#pragma once

#include "../include/film.h"

namespace luminous {
    inline namespace render {
        struct Pixel {
            float4 rgb;
            float weight_sum;
        };

        class GBufferFilm : public FilmBase {
        private:
            Pixel * _pixels;

        public:
            GBufferFilm(int2 res) : FilmBase(res) {}

            GEN_CLASS_NAME(GBufferFilm)

        };
    }
}