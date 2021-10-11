//
// Created by Zero on 11/10/2021.
//

#include "film.h"

namespace luminous {
    inline namespace render {

        void Film::add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index) {
            uint pixel_index = pixel.y * _resolution.x + pixel.x;
            color *= weight;
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(_accumulate_buffer_view[pixel_index]);
                color = lerp(a, accum_color_prev, color);
            }
            _accumulate_buffer_view[pixel_index] = make_float4(color, 1.f);
            _frame_buffer_view[pixel_index] = make_rgba(Spectrum::linear_to_srgb(color));
        }

        void Film::update() {
            auto aspect = float(_resolution.x) / float(_resolution.y);
            if (aspect > 1.f) {
                _screen_window.lower = make_float2(-aspect, -1.f);
                _screen_window.upper = make_float2(aspect, 1.f);
            } else {
                _screen_window.lower = make_float2(-1.f, -1.f / aspect);
                _screen_window.upper = make_float2(1.f, 1.f / aspect);
            }
        }
    }
}