//
// Created by Zero on 2021/3/5.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "scene_graph.h"

namespace luminous {
    inline namespace render {
        class FilmBase {
        protected:
            int2 _resolution;
            Box2f _screen_window;
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

            NDSC_XPU int2 resolution() const { return _resolution; }

            NDSC_XPU Box2f screen_window() const { return _screen_window; }

            NDSC std::string _to_string() const {
                return string_printf("resolution : %s, pixel bounds : %s", 
                                    _resolution.to_string().c_str(),
                                    _screen_window.to_string().c_str());
            }
        };

        class RGBFilm;
        class GBufferFilm;

        using lstd::Variant;

        class FilmHandle : public Variant<RGBFilm *, GBufferFilm *> {
            using Variant::Variant;
            
            NDSC_XPU int2 resolution() const;

            NDSC_XPU Box2f screen_window() const;
        };
    }
}