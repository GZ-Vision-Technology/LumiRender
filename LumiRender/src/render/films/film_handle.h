//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "film.h"
#include "rgb.h"
#include "g_buffer.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class FilmHandle : public Variant<RGBFilm, GBufferFilm> {
        public:
            using Variant::Variant;

            NDSC_XPU int2 resolution() const;

            NDSC_XPU Box2f screen_window() const;

            NDSC_XPU const char *name();

            XPU void set_accumulate_buffer(float4 *d_ptr);

            XPU void set_frame_buffer(FrameBufferType *d_ptr);

            NDSC std::string to_string() const;

            static FilmHandle create(const FilmConfig &fc);
        };
    }
}