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
            explicit BoxFilter(float2 r):FilterBase(r) {}

            LM_ND_XPU float evaluate(const float2 &p) const {
                return (std::abs(p.x) <= _radius.x && std::abs(p.y) <= _radius.y) ? 1 : 0;
            }

            FilterSample sample(const float2 &u) const {
                auto p = make_float2(lerp(u[0], -_radius.x, _radius.x), lerp(u[1], -_radius.y, _radius.y));
                return {p, 1.f};
            }

            LM_ND_XPU Float integral() const { return 4 * _radius.x * _radius.y; }

            LM_ND_XPU std::string to_string() const;

            static BoxFilter *create(const FilterConfig &config);
        }
    }
}