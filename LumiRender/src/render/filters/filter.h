//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/config.h"
#include "filter_base.h"

namespace luminous {
    inline namespace render {

        class BoxFilter;

        class GaussianFilter;

        class TriangleFilter;

        using lstd::Variant;

        class FilterHandle : public Variant<BoxFilter *, GaussianFilter *, TriangleFilter *> {
            using Variant::Variant;
        public:
            CPU_ONLY(LM_NODISCARD std::string to_string() const;)

            LM_ND_XPU float2 radius() const;

            LM_ND_XPU float integral() const;

            LM_ND_XPU float evaluate(float2 p) const;

            LM_ND_XPU FilterSample sample(float2 u) const;

            static FilterHandle create(const FilterConfig &config);
        };
    }
}