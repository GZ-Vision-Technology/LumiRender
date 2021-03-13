//
// Created by Zero on 2021/2/23.
//

#include "independent.h"

namespace luminous {
    inline namespace render {

        void LCGSampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            _rng.init(pixel);
        }

        float LCGSampler::next_1d() {
            return _rng.next();
        }

        float2 LCGSampler::next_2d() {
            return make_float2(next_1d(), next_1d());
        }

        LCGSampler *LCGSampler::create(const SamplerConfig &sc, Allocator &alloc) {
            return alloc.new_object<LCGSampler>(sc.spp);
        }

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

        PCGSampler *PCGSampler::create(const SamplerConfig &sc, Allocator &alloc) {
            return alloc.new_object<PCGSampler>(sc.spp);
        }
    }
}