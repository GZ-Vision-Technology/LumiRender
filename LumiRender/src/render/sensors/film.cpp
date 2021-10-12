//
// Created by Zero on 11/10/2021.
//

#include "film.h"

namespace luminous {
    inline namespace render {

        void Film::fill_frame_buffer(uint pixel_index) {
            switch (_fb_state) {
                case Render: {
                    float4 color = _render_buffer_view[pixel_index];
                    _frame_buffer_view[pixel_index] = make_rgba(Spectrum::linear_to_srgb(color));
                    break;
                }
                case Albedo: {
                    float4 color = _albedo_buffer_view[pixel_index];
                    _frame_buffer_view[pixel_index] = make_rgba(Spectrum::linear_to_srgb(color));
                    break;
                }
                case Normal: {
                    float3 normal = make_float3(_normal_buffer_view[pixel_index]);
                    _frame_buffer_view[pixel_index] = make_rgba((normal + 1.f) / 2.f);
                    break;
                }
                default: {
                    DCHECK(0);
                }
            }
        }

        void Film::add_render_sample(uint pixel_index, Spectrum color, float weight, uint frame_index) {
            color *= weight;
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(_render_buffer_view[pixel_index]);
                color = lerp(a, accum_color_prev, color);
            }
            _render_buffer_view[pixel_index] = make_float4(color, 1.f);
            fill_frame_buffer(pixel_index);
        }

        void Film::add_normal_sample(uint pixel_index, float3 normal, float weight, uint frame_index) {
            normal *= weight;
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(_normal_buffer_view[pixel_index]);
                normal = lerp(a, accum_color_prev, normal);
            }
            _normal_buffer_view[pixel_index] = make_float4(normal, 1.f);
        }

        void Film::add_albedo_sample(uint pixel_index, float3 albedo, float weight, uint frame_index) {
            albedo *= weight;
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(_albedo_buffer_view[pixel_index]);
                albedo = lerp(a, accum_color_prev, albedo);
            }
            _albedo_buffer_view[pixel_index] = make_float4(albedo, 1.f);
        }

        void Film::_update() {
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