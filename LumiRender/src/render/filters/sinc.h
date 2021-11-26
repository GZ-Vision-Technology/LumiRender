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
            float _tau{};
        public:
            CPU_ONLY(explicit LanczosSincFilter(const FilterConfig &config)
                    : LanczosSincFilter(config.radius, config.tau) {})

            explicit LanczosSincFilter(float2 r, float tau);

            LM_ND_XPU float evaluate(const float2 &p) const;

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })
        };
    }
}