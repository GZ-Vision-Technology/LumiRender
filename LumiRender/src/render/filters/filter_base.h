//
// Created by Zero on 2021/7/26.
//


#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/common.h"
#include "base_libs/sampling/distribution.h"

namespace luminous {
    inline namespace render {
        struct FilterSample {
            float2 p;
            float weight{};
        };

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

            void _init_distribution() {
                auto builder = Distribution2D::create_builder(_func.cbegin(), _func.row, _func.col);
                auto builder1Ds = std::move(builder.conditional_v);
                builder1Ds.push_back(builder.marginal);
                for (int y = 0; y < builder1Ds.size(); ++y) {
                    auto builder1D = builder1Ds[y];
                    for (int x = 0; x < CDF_size; ++x) {
                        CDF(x, y) = builder1D.CDF[x];
                    }
                    float *func_ptr{nullptr};
                    if (y == tab_size) {
                        func_ptr = _marginal_func;
                        for (int i = 0; i < tab_size; ++i) {
                            _marginal_func[i] = builder1D.func[i];
                        }
                    } else {
                        func_ptr = _func[y];
                    }
                    BufferView<const float> func_view(func_ptr, tab_size);
                    BufferView<const float> CDF_view(CDF[y], CDF_size);
                    _distributions[y] = Distribution1D(func_view, CDF_view, builder1D.func_integral);
                }
                _distribution2d = Distribution2D({_distributions, tab_size}, _distributions[tab_size]);
            }

        public:
            FilterSampler() = default;

            explicit FilterSampler(const Array2D<float, tab_size> &func) {
                init(func);
            }

            explicit FilterSampler(const float *func) {
                init(func);
            }

            void init(const float *func) {
                _func.fill(func);
                _init_distribution();
            }

            void init(const Array2D<float, tab_size> &func) {
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