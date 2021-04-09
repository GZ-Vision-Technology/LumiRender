//
// Created by Zero on 2021/4/9.
//

#include "uniform.h"

namespace luminous {
    inline namespace render {


        void UniformLightSampler::init(const Light *host_lights,
                                       const Light *device_lights) {
            _host_lights = host_lights;
            _device_lights = device_lights;
        }

        SampledLight UniformLightSampler::sample(float u) const {
            // todo
            return SampledLight();
        }

        SampledLight UniformLightSampler::sample(const LightSampleContext &ctx, float u) const {
            return sample(u);
        }

        float UniformLightSampler::PMF(const Light &light) const {
            return _num_lights == 0 ? 0 : 1.f / _num_lights;
        }

        float UniformLightSampler::PMF(const LightSampleContext &ctx, const Light &light) const {
            return PMF(light);
        }

        std::string UniformLightSampler::to_string() const {
            return string_printf("light sampler : %s",name());
        }

        UniformLightSampler UniformLightSampler::create(const LightSamplerConfig &config) {
            return UniformLightSampler();
        }
    } // luminous::render
} // luminous