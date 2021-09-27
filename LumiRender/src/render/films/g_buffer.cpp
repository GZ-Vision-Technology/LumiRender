//
// Created by Zero on 2021/3/5.
//

#include "g_buffer.h"

namespace luminous {
    inline namespace render {

        CPU_ONLY(GBufferFilm GBufferFilm::create(const FilmConfig &config) {
            return GBufferFilm(config.resolution);
        })
    }
}