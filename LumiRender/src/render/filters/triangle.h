//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {
        class TriangleFilter : public FilterBase {
        public:
            CPU_ONLY(explicit TriangleFilter(const FilterConfig &config) : TriangleFilter(config.radius) {})

            explicit TriangleFilter(float2 r) : FilterBase(r) {}

            LM_ND_XPU float evaluate(const float2 &p) const {
                return std::max(0.f, _radius.x - std::abs(p.x)) *
                       std::max(0.f, _radius.y - std::abs(p.y));
            }

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                return {make_float2(sample_tent(u.x, _radius.x), sample_tent(u.y, _radius.y)),1.f};
            }

            LM_ND_XPU float integral() const { return sqr(_radius.x * _radius.y); }

            GEN_STRING_FUNC({
                                return string_printf("filter type:%s, %s", type_name(this),
                                                     FilterBase::to_string().c_str());
                            })

        };
    }
}