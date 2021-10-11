//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "film_base.h"
#include "rgb.h"
#include "g_buffer.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        class Film : BASE_CLASS(lstd::Variant<RGBFilm>) {
        public:
            GEN_BASE_NAME(Film)

            REFL_CLASS(Film)

            using BaseBinder::BaseBinder;

            LM_ND_XPU uint2 resolution() const;

            LM_ND_XPU Box2f screen_window() const;

            LM_XPU void add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index = 0u);

            LM_XPU void set_resolution(uint2 res);

            LM_XPU void set_accumulate_buffer_view(BufferView<float4> buffer_view);

            LM_XPU void set_frame_buffer_view(BufferView<FrameBufferType> buffer_view);

            LM_ND_XPU float4 *accumulate_buffer_ptr();

            LM_ND_XPU FrameBufferType *frame_buffer_ptr();

            CPU_ONLY(LM_NODISCARD std::string to_string() const;)

            CPU_ONLY(static Film create(const FilmConfig &fc);)
        };
    }
}