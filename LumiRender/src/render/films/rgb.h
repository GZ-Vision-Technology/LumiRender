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

            XPU void add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index = 0u);

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("%s : {resolution :%s}", type_name<RGBFilm>(),
                                   _resolution.to_string().c_str());
            })

            CPU_ONLY(static RGBFilm create(const FilmConfig &config);)
        };
    }
}