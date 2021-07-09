//
// Created by Zero on 2021/3/5.
//

#include "rgb.h"
#include "graphics/optics/rgb.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        void RGBFilm::add_sample(uint2 pixel, Spectrum color, float weight, uint frame_index) {
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

        RGBFilm RGBFilm::create(const FilmConfig &config) {
            return RGBFilm(config.resolution);
        }
    } // luminous::render
} // luminous