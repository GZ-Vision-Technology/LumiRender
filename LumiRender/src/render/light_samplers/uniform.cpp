//
// Created by Zero on 2021/4/9.
//

#include "uniform.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        SampledLight UniformLightSampler::sample(float u) const {
            if (_lights.empty()) {
                return {};
            }
            int lightIndex = std::min<int>(u * light_num(), light_num() - 1);
            return SampledLight(&_lights[lightIndex], 1.f / light_num());
        }

        SampledLight UniformLightSampler::sample(const LightSampleContext &ctx, float u) const {
            return sample(u);
        }

        float UniformLightSampler::PMF(const Light &light) const {
            return light_num() == 0 ? 0 : 1.f / light_num();
        }

        float UniformLightSampler::PMF(const LightSampleContext &ctx, const Light &light) const {
            return PMF(light);
        }

        CPU_ONLY(UniformLightSampler UniformLightSampler::create(const LightSamplerConfig &config) {
            return UniformLightSampler();
        })
    } // luminous::render
} // luminous