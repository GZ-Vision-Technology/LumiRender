//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "filter_base.h"
#include "base_libs/math/rng.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {
        class TriangleFilter : public FilterBase {
        private:
            FilterSampler _sampler;
        public:
            CPU_ONLY(explicit TriangleFilter(const FilterConfig &config) : TriangleFilter(config.radius) {})

            explicit TriangleFilter(float2 r) : FilterBase(r) {}

            LM_ND_XPU float evaluate(const float2 &p) const {
                return std::max(0.f, _radius.x - std::abs(p.x)) *
                       std::max(0.f, _radius.y - std::abs(p.y));
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