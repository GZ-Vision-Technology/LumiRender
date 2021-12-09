//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "../include/config.h"
#include "independent.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Filter;

        class Sampler : BASE_CLASS(Variant<LCGSampler, PCGSampler>) {
        public:
            REFL_CLASS(Sampler)

        private:
            using BaseBinder::BaseBinder;
        public:
            GEN_BASE_NAME(Sampler)

            LM_NODISCARD LM_XPU int spp() const;

            LM_XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            LM_XPU SensorSample sensor_sample(uint2 p_raster, const Filter *filter);

            LM_ND_XPU int compute_dimension(int depth) const;

            GEN_TO_STRING_FUNC

            LM_ND_XPU float next_1d();

            LM_ND_XPU float2 next_2d();

            CPU_ONLY(static Sampler create(const SamplerConfig &config);)
        };
    }
}