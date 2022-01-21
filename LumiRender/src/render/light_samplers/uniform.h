//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "light_sampler_base.h"
#include "parser/config.h"

namespace luminous {
    inline namespace render {

        class UniformLightSampler : public LightSamplerBase {
        public:
            DECLARE_REFLECTION(UniformLightSampler, LightSamplerBase)

            UniformLightSampler() = default;

            CPU_ONLY(explicit UniformLightSampler(const LightSamplerConfig &config) {})

            LM_ND_XPU SampledLight sample(float u) const;

            LM_ND_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            LM_ND_XPU float PMF(const Light &light) const;

            LM_ND_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light sampler : %s", type_name(this));
                            })

        };

    } // luminous::render
} // luminous

