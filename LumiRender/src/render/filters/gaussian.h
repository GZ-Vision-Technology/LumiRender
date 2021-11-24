//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {
        class GaussianFilter : public FilterBase {
        private:
            float _exp_x{};
            float _exp_y{};
            float _sigma{};
            FilterSampler _sampler;
        public:
            CPU_ONLY(explicit GaussianFilter(const FilterConfig &config)
                    : GaussianFilter(config.radius, config.exp_x, config.exp_y, config.sigma) {})

            explicit GaussianFilter(float2 r, float exp_x, float exp_y, float sigma)
                    : FilterBase(r), _exp_x(exp_x), _exp_y(exp_y), _sigma(sigma) {}

            LM_ND_XPU float evaluate(const float2 &p) const {
                return (std::max<float>(0, gaussian(p.x, 0, _sigma) - _exp_x) *
                        std::max<float>(0, gaussian(p.y, 0, _sigma) - _exp_x));
            }

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                return {make_float2(sample_tent(u.x, _radius.x), sample_tent(u.y, _radius.y)), 1.f};
            }

            LM_ND_XPU float integral() const { return sqr(_radius.x * _radius.y); }

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })
        };
    }
}