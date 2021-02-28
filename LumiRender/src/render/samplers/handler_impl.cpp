//
// Created by Zero on 2021/2/25.
//

#include <render/include/sampler.h>

#include "lcg.h"
#include "pcg.h"

namespace luminous {
    inline namespace render {
        int SamplerHandler::spp() const {
            LUMINOUS_VAR_PTR_DISPATCH(spp)
        }

        void SamplerHandler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            LUMINOUS_VAR_PTR_DISPATCH(start_pixel_sample, pixel, sample_index, dimension)
        }

        const char *SamplerHandler::name() {
            LUMINOUS_VAR_PTR_DISPATCH(name)
        }

        float SamplerHandler::next_1d() {
            LUMINOUS_VAR_PTR_DISPATCH(next_1d)
        }

        float2 SamplerHandler::next_2d() {
            LUMINOUS_VAR_PTR_DISPATCH(next_2d)
        }

        std::string SamplerHandler::to_string() {
            LUMINOUS_VAR_PTR_DISPATCH(to_string)
        }

        SamplerHandler SamplerHandler::create(const Config &config) {
            const SamplerConfig &sampler_config = (SamplerConfig &) config;
            auto creator = GET_CREATOR(sampler_config.type);
            if (sampler_config.type == "LCGSampler") {
                auto sampler =(LCGSampler *) creator(config);
                return SamplerHandler(sampler);
            } else if (sampler_config.type == "PCGSampler") {
                auto sampler =(PCGSampler *) creator(config);
                return SamplerHandler(sampler);
            } else {
                LUMINOUS_ERROR("unknow sampler type:", sampler_config.type)
            }

        }
    }
}