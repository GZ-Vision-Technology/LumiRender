//
// Created by Zero on 2021/1/29.
//
#ifndef __CUDACC__
#include <random>
#endif

#include "base_libs/math/common.h"
#include "independent.h"

namespace luminous {
    inline namespace render {

#ifndef __CUDACC__
        static const auto rd_dev_seed = []() -> std::default_random_engine::result_type {
            std::random_device rd_dev;
            return rd_dev();
        }();
        static std::default_random_engine random(rd_dev_seed);
        static std::uniform_real_distribution<float> dis2(0.f, 1.f);
#endif

        void DebugSampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            _rng.init(pixel * uint(sample_index));
        }

        int DebugSampler::compute_dimension(int depth) const {
            return 5 + depth * 7;
        }

        float DebugSampler::next_1d() {
#ifndef __CUDACC__
            auto ret = dis2(random);
            return ret;
#else
            return _rng.next();
#endif
        }

        float2 DebugSampler::next_2d() {
            float x = next_1d();
            float y = next_1d();
            return make_float2(x, y);
        }

        void PCGSampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            _rng.set_sequence(Hash(pixel, 0));
            _rng.advance(sample_index * 65536ull + dimension);
        }

        int PCGSampler::compute_dimension(int depth) const {
            return 5 + depth * 7;
        }

        float PCGSampler::next_1d() {
            return _rng.uniform<float>();
        }

        float2 PCGSampler::next_2d() {
            float x = next_1d();
            float y = next_1d();
            return make_float2(x, y);
        }
    }
}