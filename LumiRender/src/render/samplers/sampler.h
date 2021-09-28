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
        class Sampler : BASE_CLASS(Variant<LCGSampler, PCGSampler>) {
        public:
            REFL_CLASS(Sampler)
        private:
            using BaseBinder::BaseBinder;
        public:
            GEN_BASE_NAME(Sampler)

            LM_NODISCARD XPU int spp() const;

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            XPU SensorSample sensor_sample(uint2 p_raster);

            GEN_TO_STRING_FUNC

            NDSC_XPU float next_1d();

            NDSC_XPU float2 next_2d();

            CPU_ONLY(static Sampler create(const SamplerConfig &config);)
        };

    }
}