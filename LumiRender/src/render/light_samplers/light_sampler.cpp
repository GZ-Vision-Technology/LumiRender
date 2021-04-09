//
// Created by Zero on 2021/4/9.
//

#include "light_sampler.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        void LightSampler::init(const Light *host_lights, const Light *device_lights) {
            LUMINOUS_VAR_DISPATCH(init, host_lights, device_lights);
        }

        SampledLight LightSampler::sample(float u) const {
            LUMINOUS_VAR_DISPATCH(sample, u);
        }

        SampledLight LightSampler::sample(const LightSampleContext &ctx, float u) const {
            LUMINOUS_VAR_DISPATCH(sample, ctx, u);
        }

        float LightSampler::PMF(const Light &light) const {
            LUMINOUS_VAR_DISPATCH(PMF, light);
        }

        float LightSampler::PMF(const LightSampleContext &ctx, const Light &light) const {
            LUMINOUS_VAR_DISPATCH(PMF, ctx, light);
        }

        std::string LightSampler::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string);
        }

        LightSampler LightSampler::create(const LightSamplerConfig &config) {
            return detail::create<LightSampler>(config);
        }
    } // luminous::render
} // luminous