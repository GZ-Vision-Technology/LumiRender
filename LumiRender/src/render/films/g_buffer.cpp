//
// Created by Zero on 2021/3/5.
//

#include "g_buffer.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        GBufferFilm GBufferFilm::create(const FilmConfig &config) {
            return GBufferFilm(config.resolution);
        }


    }
}