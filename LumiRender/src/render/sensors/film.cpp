//
// Created by Zero on 11/10/2021.
//

#include "film.h"

namespace luminous {
    inline namespace render {

    template<class Td, std::enable_if_t<std::is_same_v<Td, uint32_t> || std::is_same_v<std::remove_reference_t<Td>, float4>, int> = 0>
    LM_XPU static inline void fill_frame_buffer_pixel_loc(const float3 &src, Td &dest) {
        if constexpr(std::is_same_v<Td, float4>)
            dest = make_float4(src, 1.0f);
        else
            dest = make_rgba(src);
    }

        void Film::fill_frame_buffer(uint pixel_index) {
            switch (_fb_state) {
                case Render: {
                    float4 val = _render_buffer_view[pixel_index];
                    if (val.w == 0) {
                        return;
                    }
                    float3 color = make_float3(val);
                    // float4 framebuffer lead performance reduction about 10%
                    fill_frame_buffer_pixel_loc(Spectrum::linear_to_srgb(color), _frame_buffer_view[pixel_index]);
                    break;
                }
                case Albedo: {
                    float4 val = _albedo_buffer_view[pixel_index];
                    if (val.w == 0) {
                        return;
                    }
                    float3 color = make_float3(val);
                    fill_frame_buffer_pixel_loc(Spectrum::linear_to_srgb(color), _frame_buffer_view[pixel_index]);
                    break;
                }
                case Normal: {
                    float4 val = _normal_buffer_view[pixel_index];
                    if (val.w == 0) {
                        return;
                    }
                    float3 normal = make_float3(val);
                    fill_frame_buffer_pixel_loc((normal + 1.f) / 2.f, _frame_buffer_view[pixel_index]);
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
            DCHECK(!has_invalid(val));
            if (frame_index > 0) {
                const float a = 1.0f / static_cast<float>(frame_index + 1);
                const float3 accum_color_prev = make_float3(buffer_view[pixel_index]);
                val = lerp(a, accum_color_prev, val);
            }
            buffer_view[pixel_index] = make_float4(val, 1.f);
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