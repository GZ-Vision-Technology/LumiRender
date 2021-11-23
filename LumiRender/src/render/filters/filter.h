//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/config.h"
#include "box.h"
#include "triangle.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Filter : BASE_CLASS(Variant<BoxFilter, TriangleFilter>) {
        private:
            using BaseBinder::BaseBinder;
        public:
            GEN_BASE_NAME(Sampler)

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                LUMINOUS_VAR_DISPATCH(sample, u);
            }

            LM_ND_XPU float integral() const {
                LUMINOUS_VAR_DISPATCH(integral);
            }

            CPU_ONLY(static Filter create(const FilterConfig &config) {
                return detail::create<Filter>(config);
            })
        };
    }
}