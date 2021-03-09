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
            Box2i _pixel_bounds;
        public:
            FilmBase(int2 res)
                : _resolution(res) {
                _pixel_bounds = Box2i(make_int2(0,0), _resolution);
            }

            NDSC_XPU int2 resolution() const { return _resolution; }

            NDSC_XPU Box2i pixel_bounds() const { return _pixelBounds; }

            NDSC std::string _to_string() const {
                return string_printf("resolution : %s, pixel bounds : %s", 
                                    _resolution.to_string().c_str(),
                                    _pixelBounds.to_string().c_str());
            }
        };

        class RGBFilm;
        class GBufferFilm;

        using lstd::Variant;

        class FilmHandle : public Variant<RGBFilm *, GBufferFilm *> {
            using Variant::Variant;
            
        } 
    }
}