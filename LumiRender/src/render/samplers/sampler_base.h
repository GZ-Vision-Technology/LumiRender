//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "graphics/geometry/common.h"

namespace luminous {
    inline namespace render {

        struct SensorSample {
            float2 p_film;
            float2 p_lens;
            float time;
            float weight{1.f};
            NDSC std::string to_string() const {
                return string_printf("p_film: %s, p_lens : %s, time: %f, weight : %f",
                                     p_film.to_string().c_str(),
                                     p_lens.to_string().c_str(),
                                     time, weight);
            }
        };

        class SamplerBase {
        protected:
            int _spp;
        public:
            XPU explicit SamplerBase(int spp = 1) : _spp(spp) {}

            NDSC int spp() const { return _spp; }
        };
    }
}