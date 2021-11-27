//
// Created by Zero on 11/10/2021.
//

#include "film.h"

namespace luminous {
    inline namespace render {

        void Film::fill_frame_buffer(uint pixel_index) {
            switch (_fb_state) {
                case Render: {
                    float4 val = _render_buffer_view[pixel_index];
                    if (val.w == 0) {
                        return;
                    }
                    float3 color = make_float3(val) / val.w;
                    _frame_buffer_view[pixel_index] = make_rgba(Spectrum::linear_to_srgb(color));
                    break;
                }
                case Albedo: {
                    float4 val = _albedo_buffer_view[pixel_index];
                    if (val.w == 0) {
                        return;
                    }
                    float3 color = make_float3(val) / val.w;
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

        void Film::fill_buffer(uint pixel_index, float3 val, float weight,
                               uint frame_index, BufferView<float4> buffer_view) {
            val = val * weight;
            if (frame_index == 0) {
                buffer_view[pixel_index] = make_float4(val, weight);
            } else {
                float pre_weight_sum = buffer_view[pixel_index].w;
                const float3 accum_val_prev = make_float3(buffer_view[pixel_index]);
                buffer_view[pixel_index] = make_float4(val + accum_val_prev, pre_weight_sum + weight);
            }
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