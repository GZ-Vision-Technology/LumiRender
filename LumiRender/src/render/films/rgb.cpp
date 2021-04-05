//
// Created by Zero on 2021/3/5.
//

#include "rgb.h"
#include "graphics/optics/rgb.h"

namespace luminous {
    inline namespace render {

        void RGBFilm::add_sample(float2 p_film, float3 color, float weight, uint frame_index) {
            auto p = make_int2(p_film);
            uint pixel_index = p.y * _resolution.x + p.x;
//            _d_frame_buffer[pixel_index] = make_rgba(color * weight);
            color *= weight;
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(_d_accumulate_buffer[pixel_index]);
                color = lerp(a, accum_color_prev, color);
            }
            _d_accumulate_buffer[pixel_index] = make_float4(color, 1.f);
            _d_frame_buffer[pixel_index] = make_rgba(color);
        }

        std::string RGBFilm::to_string() const {
            return string_printf("%s : {resolution :%s}", name(),
                                 _resolution.to_string().c_str());
        }

        RGBFilm RGBFilm::create(const FilmConfig &config) {
            return RGBFilm(config.resolution);
        }
    } // luminous::render
} // luminous