//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "../lights/light_handle.h"

namespace luminous {
    inline namespace render {

        class UniformLightSampler {
        private:

        public:
            NDSC_XPU SampledLight sample(float u) const;

            NDSC_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU float PMF(const LightHandle &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, float u) const;

            NDSC std::string to_string() const;
        };

    } // luminous::render
} // luminous

