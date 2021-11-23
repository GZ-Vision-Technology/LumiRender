//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "filter_base.h"
#include "base_libs/math/rng.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {
        class BoxFilter : public FilterBase {
        public:
            CPU_ONLY(explicit BoxFilter(const FilterConfig &config) : BoxFilter(config.radius) {})

            explicit BoxFilter(float2 r) : FilterBase(r) {}

            LM_ND_XPU float evaluate(const float2 &p) const {
                return (std::abs(p.x) <= _radius.x && std::abs(p.y) <= _radius.y) ? 1 : 0;
            }

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                auto p = make_float2(lerp(u[0], -_radius.x, _radius.x), lerp(u[1], -_radius.y, _radius.y));
                return {p, 1.f};
            }

            LM_ND_XPU float integral() const { return 4 * _radius.x * _radius.y; }

            GEN_STRING_FUNC({
                return string_printf("filter type:%s, %s", type_name(this), FilterBase::to_string().c_str());
            })

        };
    }
}