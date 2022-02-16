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
                func[i] = abs(val);
                _lut(x, y) = val;
            }
            auto sum = std::reduce(_lut.cbegin(), _lut.cend(), 0.0);
            auto integral = sum / len;
            std::transform(_lut.cbegin(), _lut.cend(), _lut.begin(), [&](auto val) {
                return val / integral;
            });
            init(func.data());
        }

        void FilterSampler::init(const float *func) {
            _distribution2d = create_static_distrib2d<tab_size, tab_size>(func);
        }

        FilterSample FilterSampler::sample(float2 u) const {
            u = u * 2.f - make_float2(1.f);
            float PDF = 0;
            int2 offset{};
            float2 p = _distribution2d.sample_continuous(abs(u), &PDF, &offset);
            p = p * sign(u);
            return FilterSample{p, _lut(offset) / PDF};
        }
    }
}