//
// Created by Zero on 11/10/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/config.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {

        struct Pixel {
            float4 rgb{};
            float weight_sum{};
        };

        class Film : BASE_CLASS() {
        public:
            REFL_CLASS(Film)

        protected:
            uint2 _resolution;
            Box2f _screen_window;
            BufferView<float4> _accumulate_buffer_view;
            BufferView<FrameBufferType> _frame_buffer_view;

            LM_XPU void update();

        public:
            CPU_ONLY(explicit Film(const FilmConfig &config) : Film(config.resolution) {})

            Film() = default;

            explicit Film(uint2 res)
                    : _resolution(res) {
                update();
            }

            LM_XPU void add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index = 0u);

            LM_XPU void set_resolution(uint2 resolution) {
                _resolution = resolution;
                update();
            }

            LM_XPU void set_accumulate_buffer_view(BufferView<float4> buffer_view) {
                _accumulate_buffer_view = buffer_view;
            }

            LM_XPU void set_frame_buffer_view(BufferView<FrameBufferType> buffer_view) {
                _frame_buffer_view = buffer_view;
            }

            LM_ND_XPU float4 *accumulate_buffer_ptr() {
                return _accumulate_buffer_view.ptr();
            }

            LM_ND_XPU FrameBufferType *frame_buffer_ptr() {
                return _frame_buffer_view.ptr();
            }

            LM_ND_XPU uint2 resolution() const { return _resolution; }

            LM_ND_XPU Box2f screen_window() const { return _screen_window; }

            GEN_STRING_FUNC({
                                return string_printf("resolution : %s, pixel bounds : %s",
                                                     _resolution.to_string().c_str(),
                                                     _screen_window.to_string().c_str());
                            })
        };
    }
}