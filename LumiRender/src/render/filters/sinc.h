//
// Created by Zero on 25/11/2021.
//


#pragma once

#include "filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "render/include/config.h"
#include "filter_sampler.h"

namespace luminous {
    inline namespace render {
        class LanczosSincFilter : public FittedFilter {
        private:
            float tau{};
        public:
            CPU_ONLY(explicit LanczosSincFilter(const FilterConfig &config)
                    : GaussianFilter(config.radius, config.sigma) {})

            explicit LanczosSincFilter(float2 r, float sigma);

            LM_ND_XPU float evaluate(const float2 &p) const {
                return (std::max<float>(0, gaussian(p.x, 0, _sigma) - _exp_x) *
                        std::max<float>(0, gaussian(p.y, 0, _sigma) - _exp_y));
            }

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })
        };
    }
}