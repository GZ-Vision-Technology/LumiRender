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
            float filter_weight{1.f};
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