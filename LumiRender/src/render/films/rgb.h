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
        }

        class RGBFilm : public FilmBase {
        private:
            Pixel * _pixels;

        public:
            RGBFilm(int2 res) : FilmBase(res) {}

            GEN_CLASS_NAME(RGBFilm)

            XPU void add_sample(float2 p_film, float3 color, float weight);
        };
    }
}