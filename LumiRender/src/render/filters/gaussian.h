//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "render/include/config.h"
#include "filter_sampler.h"

namespace luminous {
    inline namespace render {

        class Filter;

        class GaussianFilter : public FilterBase {
        private:
            float _exp_x{};
            float _exp_y{};
            float _sigma{};
            FilterSampler _sampler;
        public:
            CPU_ONLY(explicit GaussianFilter(const FilterConfig &config)
                    : GaussianFilter(config.radius, config.sigma) {})

            explicit GaussianFilter(float2 r, float sigma);

            LM_ND_XPU float evaluate(const float2 &p) const {
                return (std::max<float>(0, gaussian(p.x, 0, _sigma) - _exp_x) *
                        std::max<float>(0, gaussian(p.y, 0, _sigma) - _exp_y));
            }

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                return _sampler.sample(u);
            }

            LM_ND_XPU float integral() const { return sqr(_radius.x * _radius.y); }

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })
        };
    }
}