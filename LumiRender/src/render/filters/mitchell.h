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
        class MitchellFilter : public FittedFilter {
        private:
            float b;
            float c;

            LM_ND_XPU float mitchell_1d(float x) const {
                x = std::abs(x);
                if (x <= 1)
                    return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x +
                            (6 - 2 * b)) *
                           (1.f / 6.f);
                else if (x <= 2)
                    return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x +
                            (-12 * b - 48 * c) * x + (8 * b + 24 * c)) *
                           (1.f / 6.f);
                else
                    return 0;
            }

        public:
            CPU_ONLY(explicit MitchellFilter(const FilterConfig &config)
                    : MitchellFilter(config.radius, config.b, config.c) {})

            explicit MitchellFilter(float2 r, float b = 1.f / 3.f, float c = 1.f / 3.f);

            LM_ND_XPU float evaluate(const float2 &p) const;

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })
        };
    }
}