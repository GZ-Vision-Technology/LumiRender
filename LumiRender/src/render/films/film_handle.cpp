//
// Created by Zero on 2021/3/5.
//

#include "film_handle.h"

namespace luminous {
    inline namespace render {
        int2 FilmHandle::resolution() const {
            LUMINOUS_VAR_DISPATCH(resolution);
        }

        Box2f FilmHandle::screen_window() const {
            LUMINOUS_VAR_DISPATCH(screen_window);
        }
    }
}