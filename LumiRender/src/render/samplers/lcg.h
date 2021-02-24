//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "../include/sampler.h"
#include "graphics/math/rng.h"
#include "../include/scene_graph.h"

namespace luminous {
    inline namespace render {
        class LCGSampler : public SamplerBase {
        private:
            LCG<> _rng;
        public:
            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            GEN_CLASS_NAME(LCGSampler)

            NDSC XPU float next_1d();

            NDSC XPU float2 next_2d();

            NDSC std::string to_string() const {
                return string_printf("%s:{spp=%d}", name(), spp());
            }
        };


    }
}