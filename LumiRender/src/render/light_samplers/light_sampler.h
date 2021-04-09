//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "uniform.h"
#include "power.h"
#include "bvh.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class LightSampler : public Variant<UniformLightSampler> {
        public:
            using Variant::Variant;

            GEN_BASE_NAME(LightSampler)

            void init(const Light * host_lights, const Light *device_lights);

            NDSC_XPU SampledLight sample(float u) const;

            NDSC_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU float PMF(const Light &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            NDSC std::string to_string() const;

            static LightSampler create(const LightSamplerConfig &config);
        };

    } // luminous::render
} // luminous