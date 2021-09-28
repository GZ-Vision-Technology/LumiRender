//
// Created by Zero on 2021/3/5.
//


#pragma once


#include "film_base.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {

        class RGBFilm : BASE_CLASS(FilmBase), public Creator<RGBFilm> {
        public:
            REFL_CLASS(RGBFilm)

            CPU_ONLY(explicit RGBFilm(const FilmConfig &config) : RGBFilm(config.resolution) {})

            XPU explicit RGBFilm(uint2 res) : BaseBinder<FilmBase>(res) {}

            XPU void add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index = 0u);

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s : {resolution :%s}", type_name<RGBFilm>(),
                                                   _resolution.to_string().c_str());
                            })
        };
    }
}