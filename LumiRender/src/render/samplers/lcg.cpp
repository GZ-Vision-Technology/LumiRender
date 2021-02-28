//
// Created by Zero on 2021/2/23.
//

#include "lcg.h"
#include "../include/class_factory.h"

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

        IObject * create(const Config &config) {
            return new LCGSampler();
        }

        REGISTER(LCGSampler, create);
    }
}