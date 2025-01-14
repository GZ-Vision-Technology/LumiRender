//
// Created by Zero on 24/11/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/common.h"
#include "base_libs/sampling/distribution.h"

namespace luminous {
    inline namespace render {

        class Filter;

        struct FilterSample {
            float2 p;
            float weight{};
        };

        class FilterSampler {
        public:
            static constexpr int tab_size = 20;
            static constexpr int CDF_size = tab_size + 1;
            using Distrib = StaticDistribution2D<tab_size, tab_size>;
            using array_type = Array2D<float, tab_size, tab_size>;
        private:
            Distrib _distribution2d;
            array_type _lut{};
        public:
            FilterSampler() = default;

            explicit FilterSampler(const Filter *filter) {
                init(filter);
            }

            explicit FilterSampler(const float *func) {
                init(func);
            }

            void init(const Filter *filter);

            void init(const float *func);

            LM_ND_XPU FilterSample sample(float2 u) const;
        };
    }
}