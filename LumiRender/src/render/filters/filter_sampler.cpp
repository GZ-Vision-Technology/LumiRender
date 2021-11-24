//
// Created by Zero on 24/11/2021.
//

#include "filter_sampler.h"
#include "filter.h"

namespace luminous {
    inline namespace render {

        void FilterSampler::init(const Filter *filter) {
            int len = sqr(tab_size);
            std::vector<float> func;
            func.resize(len);
            float2 r = filter->radius();
            for (int i = 0; i < len; ++i) {
                int x = i % tab_size;
                int y = i / tab_size;
                float2 p = make_float2((x + 0.5f) / tab_size * r.x,
                                       (y + 0.5f) / tab_size * r.y);
                float val = filter->evaluate(p);
                func[i] = val;
            }
            init(func.data());
        }

        void FilterSampler::_init_distribution() {
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
    }
}