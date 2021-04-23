//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "light_sampler_base.h"

namespace luminous {
    inline namespace render {

        class UniformLightSampler : public LightSamplerBase {
        public:
            UniformLightSampler() = default;

            NDSC_XPU SampledLight sample(float u) const;

            NDSC_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU float PMF(const Light &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            NDSC std::string to_string() const;

            static UniformLightSampler create(const LightSamplerConfig &config);
        };

    } // luminous::render
} // luminous

