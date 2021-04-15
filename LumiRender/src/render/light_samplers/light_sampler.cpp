//
// Created by Zero on 2021/4/9.
//

#include "light_sampler.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        const char *LightSampler::name() const {
            LUMINOUS_VAR_DISPATCH(name);
        }

        void LightSampler::set_lights(BufferView<const Light> lights) {
            LUMINOUS_VAR_DISPATCH(set_lights, lights);
        }

        size_t LightSampler::light_num() {
            LUMINOUS_VAR_DISPATCH(light_num);
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
#ifdef IS_GPU_CODE
            LUMINOUS_ERROR("device disable to_string");
#else
            LUMINOUS_VAR_DISPATCH(to_string);
#endif
        }

        LightSampler LightSampler::create(const LightSamplerConfig &config) {
            return detail::create<LightSampler>(config);
        }

        BufferView<const Light> LightSampler::lights() const {
            LUMINOUS_VAR_DISPATCH(lights);
        }

    } // luminous::render
} // luminous