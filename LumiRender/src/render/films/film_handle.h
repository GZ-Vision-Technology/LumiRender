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
        class FilmHandle : public Variant<RGBFilm> {
        public:
            using Variant::Variant;

            NDSC_XPU uint2 resolution() const;

            NDSC_XPU Box2f screen_window() const;

            NDSC_XPU const char *name();

            XPU void add_sample(float2 p_film, float3 color, float weight, uint frame_index = 0u);

            XPU void set_resolution(uint2 res);

            XPU void set_accumulate_buffer(float4 *d_ptr);

            XPU void set_frame_buffer(FrameBufferType *d_ptr);

            NDSC std::string to_string() const;

            static FilmHandle create(const FilmConfig &fc);
        };
    }
}