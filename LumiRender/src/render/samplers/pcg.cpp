//
// Created by Zero on 2021/2/23.
//


#include "pcg.h"

namespace luminous {
    inline namespace render {

        void PCGSampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            uint64_t idx = (pixel.x + pixel.y) * pixel.x;
            _rng.pcg32_init(idx);
        }

        float PCGSampler::next_1d() {
            return (float) _rng.uniform_float();
        }

        float2 PCGSampler::next_2d() {
            return make_float2(next_1d(), next_1d());
        }

        IObject *create_PCGSampler(const Config &config) {
            const SamplerConfig &sc = (SamplerConfig &) config;
            return new PCGSampler(sc.spp);
        }

        REGISTER(PCGSampler, create_PCGSampler);
    }
}