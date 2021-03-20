//
// Created by Zero on 2021/3/5.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "../include/scene_graph.h"

namespace luminous {
    inline namespace render {
        struct Pixel {
            float4 rgb;
            float weight_sum;
        };

        using FrameBufferType = uchar4;

        class FilmBase {
        protected:
            int2 _resolution;
            Box2f _screen_window;
            float4 *_d_accumulate_buffer{};
            FrameBufferType *_d_frame_buffer{};
        public:
            FilmBase(int2 res)
                : _resolution(res) {
                auto aspect = float(res.x) / float(res.y);
                if (aspect > 1.f) {
                    _screen_window.lower = make_float2(-aspect, -1.f);
                    _screen_window.upper = make_float2(aspect, 1.f);
                } else {
                    _screen_window.lower = make_float2(-1.f, -1.f / aspect);
                    _screen_window.upper = make_float2(1.f, 1.f / aspect);
                }
            }

            XPU void set_accumulate_buffer(float4 *d_ptr) {
                _d_accumulate_buffer = d_ptr;
            }

            XPU void set_frame_buffer(FrameBufferType *d_ptr) {
                _d_frame_buffer = d_ptr;
            }

            NDSC_XPU int2 resolution() const { return _resolution; }

            NDSC_XPU Box2f screen_window() const { return _screen_window; }

            NDSC std::string _to_string() const {
                return string_printf("resolution : %s, pixel bounds : %s", 
                                    _resolution.to_string().c_str(),
                                    _screen_window.to_string().c_str());
            }
        };
    }
}