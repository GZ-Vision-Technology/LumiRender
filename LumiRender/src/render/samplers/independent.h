//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/math/rng.h"
#include "../include/config.h"
#include "sampler_base.h"

namespace luminous {
    inline namespace render {

        class LCGSampler : public SamplerBase {
        private:
            LCG<> _rng;
        public:
            XPU explicit LCGSampler(int spp = 1) : SamplerBase(spp) {}

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            _NODISCARD XPU float next_1d();

            _NODISCARD XPU float2 next_2d();

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("%s:{spp=%d}", type_name(this), spp())
            })

            CPU_ONLY(static LCGSampler create(const SamplerConfig &config);)
        };

        class PCGSampler : public SamplerBase {
        private:
            RNG _rng;
            int _seed;
        public:
            XPU explicit PCGSampler(int spp = 1) : SamplerBase(spp) {}

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            _NODISCARD XPU float next_1d();

            _NODISCARD XPU float2 next_2d();

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("%s:{spp=%d}", type_name(this), spp())
            })

            CPU_ONLY(static PCGSampler create(const SamplerConfig &config);)
        };
    }
}