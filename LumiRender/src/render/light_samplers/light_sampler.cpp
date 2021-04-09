//
// Created by Zero on 2021/4/9.
//

#include "light_sampler.h"

namespace luminous {
    inline namespace render {

        void LightSampler::init(const LightHandle *host_lights, const LightHandle *device_lights) {
            LUMINOUS_VAR_DISPATCH(init, host_lights, device_lights);
        }

        SampledLight LightSampler::sample(float u) const {
            LUMINOUS_VAR_DISPATCH(sample, u);
        }

        SampledLight LightSampler::sample(const LightSampleContext &ctx, float u) const {
            LUMINOUS_VAR_DISPATCH(sample, ctx, u);
        }

        float LightSampler::PMF(const LightHandle &light) const {
            LUMINOUS_VAR_DISPATCH(PMF, light);
        }

        float LightSampler::PMF(const LightSampleContext &ctx, const LightHandle &light) const {
            LUMINOUS_VAR_DISPATCH(PMF, ctx, light);
        }

        std::string LightSampler::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string);
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC LightSampler create_light_sampler(const LightSamplerConfig &config) {
                using Class = std::remove_pointer_t<std::tuple_element_t<current_index, LightSampler::TypeTuple>>;
                if (Class::name() == config.type) {
                    return LightSampler(Class::create(config));
                }
                return create_light_sampler<current_index + 1>(config);
            }

            template<>
            NDSC LightSampler create_light_sampler<std::tuple_size_v<LightSampler::TypeTuple>>(const LightSamplerConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        LightSampler LightSampler::create(const LightSamplerConfig &config) {
            return detail::create_light_sampler<0>(config);
        }
    } // luminous::render
} // luminous