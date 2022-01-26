//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "parser/config.h"
#include "box.h"
#include "triangle.h"
#include "sinc.h"
#include "gaussian.h"
#include "mitchell.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Filter : public Variant<BoxFilter, TriangleFilter, GaussianFilter,
                                          LanczosSincFilter, MitchellFilter> {

          DECLARE_REFLECTION(Filter, Variant)

        private:
            using Variant::Variant;
        public:
            LM_ND_XPU float2 radius() const {
                LUMINOUS_VAR_DISPATCH(radius);
            }

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                LUMINOUS_VAR_DISPATCH(sample, u);
            }

            LM_ND_XPU float evaluate(const float2 &p) const {
                LUMINOUS_VAR_DISPATCH(evaluate, p);
            }

            CPU_ONLY(static Filter create(const FilterConfig &config) {
                return detail::create<Filter>(config);
            })
        };
    }
}