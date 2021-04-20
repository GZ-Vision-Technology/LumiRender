//
// Created by Zero on 2021/3/5.
//

#include "g_buffer.h"

namespace luminous {
    inline namespace render {

        GBufferFilm GBufferFilm::create(const FilmConfig &config) {
            return GBufferFilm(config.resolution);
        }

        std::string GBufferFilm::to_string() const {
            LUMINOUS_TO_STRING("%s : {resolution :%s}", name().c_str(),
                                 _resolution.to_string().c_str());
        }
    }
}