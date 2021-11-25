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

        template<typename value_type>
        LM_XPU void fill_buffer(uint pixel_index, value_type val, float weight,
                                uint frame_index, BufferView<float4> buffer_view) {
            if (frame_index == 0) {
                buffer_view[pixel_index] = make_float4(val * weight, weight);
            } else {
                float pre_weight_sum = buffer_view[pixel_index].w;
                const float3 accum_color_prev = make_float3(buffer_view[pixel_index]);
                float t = weight / (pre_weight_sum + weight);
                val = lerp(t, accum_color_prev, val);
                buffer_view[pixel_index] = make_float4(val, pre_weight_sum + weight);
            }
        }

        void Film::add_render_sample(uint pixel_index, Spectrum color, float weight, uint frame_index) {
            fill_buffer(pixel_index, color, weight, frame_index, _render_buffer_view);
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