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


    }
}