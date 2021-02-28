//
// Created by Zero on 2021/2/23.
//


#pragma once

#include "../include/sampler.h"
#include "graphics/math/rng.h"

namespace luminous {
    inline namespace render {
        class PCGSampler : public SamplerBase {
        private:
            PCG _rng;
        public:
            explicit PCGSampler(int spp = 1) : SamplerBase(spp) {}

            GEN_CLASS_NAME(PCGSampler)

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            NDSC XPU float next_1d();

            NDSC XPU float2 next_2d();

            NDSC std::string to_string() const {
                return string_printf("%s:{spp=%d}", name(), spp());
            }
        };
    }
}