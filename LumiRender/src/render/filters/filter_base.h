//
// Created by Zero on 2021/7/26.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/sampling/distribution.h"

namespace luminous {
    inline namespace render {
        struct FilterSample {
            float2 p;
            float weight{};
        };

        class FilterSampler {
        public:
            static constexpr int size = 10;
            static constexpr int CDF_size = size + 1;
        private:
            Array2D<float, size> _func;
            Array2D<float, CDF_size> CDF;
            Distribution1D _distributions[size + 1];
            Distribution2D _distribution2d{};
        private:

            void _init_distribution() {

            }

        public:
            FilterSampler() = default;

            explicit FilterSampler(const Array2D<float, size> &func) {
                init(func);
            }

            void init(const float *func) {
                _func.fill(func);
                _init_distribution();
            }

            void init(const Array2D<float, size> &func) {
                _func = func;
                _init_distribution();
            }

            LM_ND_XPU FilterSample sample(float2 u) {
                float PDF;
                int2 pi;
                float2 p = _distribution2d.sample_continuous(u, &PDF, &pi);
                return {p, _func(pi) / PDF};
            }
        };

        struct FilterBase {
        protected:
            const float2 _radius;
        public:
            explicit FilterBase(const float2 r) : _radius(r) {}

            LM_ND_XPU float2 radius() {
                return _radius;
            }
        };
    }
}