//
// Created by Zero on 2021/1/29.
//


#include "sampler.h"
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

        LCGSampler LCGSampler::create(const SamplerConfig &sc) {
            return LCGSampler(sc.spp);
        }

        std::string LCGSampler::to_string() const {
            LUMINOUS_TO_STRING("%s:{spp=%d}", name().c_str(), spp())
        }

        void PCGSampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            _rng.set_sequence((pixel.x + pixel.y * 65536) | (uint64_t(_seed) << 32));
            _rng.advance(sample_index * 65536 + dimension);
        }

        float PCGSampler::next_1d() {
            return _rng.uniform<float>();
        }

        float2 PCGSampler::next_2d() {
            return make_float2(next_1d(), next_1d());
        }

        PCGSampler PCGSampler::create(const SamplerConfig &sc) {
            return PCGSampler(sc.spp);
        }

        std::string PCGSampler::to_string() const {
            LUMINOUS_TO_STRING("%s:{spp=%d}", name().c_str(), spp())
        }
    }
}