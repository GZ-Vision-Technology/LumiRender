//
// Created by Zero on 2021/3/5.
//

#include "../include/film.h"
#include "rgb.h"
#include "g_buffer.h"

namespace luminous {
    inline namespace render {
        int2 FilmHandle::resolution() const {
            LUMINOUS_VAR_PTR_DISPATCH(resolution);
        }

        Box2f FilmHandle::screen_window() const {
            LUMINOUS_VAR_PTR_DISPATCH(screen_window);
        }
    }
}