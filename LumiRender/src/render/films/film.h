//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "film_base.h"
#include "rgb.h"
#include "g_buffer.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class Film : public Variant<RGBFilm> {
        public:
            GEN_BASE_NAME(Film)

            using Variant::Variant;

            NDSC_XPU uint2 resolution() const;

            NDSC_XPU Box2f screen_window() const;

            XPU void add_sample(uint2 pixel, float3 color, float weight, uint frame_index = 0u);

            XPU void set_resolution(uint2 res);

            XPU void set_accumulate_buffer_view(BufferView<float4> buffer_view);

            XPU void set_frame_buffer_view(BufferView<FrameBufferType> buffer_view);

            GEN_NAME_AND_TO_STRING_FUNC

            static Film create(const FilmConfig &fc);
        };
    }
}