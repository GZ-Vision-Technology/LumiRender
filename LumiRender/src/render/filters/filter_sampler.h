//
// Created by Zero on 24/11/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/common.h"
#include "base_libs/sampling/distribution.h"
#include "filter_base.h"

namespace luminous {
    inline namespace render {

        class Filter;

        class FilterSampler {
        public:
            static constexpr int tab_size = 10;
            static constexpr int CDF_size = tab_size + 1;
        private:
            Array2D<float, tab_size> _func{};
            float _marginal_func[tab_size]{};
            Array2D<float, CDF_size> CDF{};
            Distribution1D _distributions[CDF_size];
            Distribution2D _distribution2d{};
        private:

            void _init_distribution();

        public:
            FilterSampler() = default;

            explicit FilterSampler(const Filter *filter) {
                init(filter);
            }

            explicit FilterSampler(const Array2D<float, tab_size> &func) {
                init(func);
            }

            explicit FilterSampler(const float *func) {
                init(func);
            }

            void init(const Filter *filter);

            void init(const float *func) {
                _func.fill(func);
                _init_distribution();
            }

            void init(const Array2D<float, tab_size> &func) {
                _func = func;
                _init_distribution();
            }

            LM_ND_XPU FilterSample sample(float2 u) const {
                float PDF;
                int2 pi;
                float2 p = _distribution2d.sample_continuous(u, &PDF, &pi);
                return {p, _func(pi) / PDF};
            }
        };
    }
}