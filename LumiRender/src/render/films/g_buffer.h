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

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("%s : {resolution :%s}", type_name<GBufferFilm>(),
                                   _resolution.to_string().c_str());
            })

            static GBufferFilm create(const FilmConfig &config);
        };
    }
}