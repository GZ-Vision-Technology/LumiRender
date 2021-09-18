//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "base_libs/math/common.h"
#include "base_libs/lstd/lstd.h"
#include "gaussian.h"
#include "triangle.h"
#include "box.h"
#include "config.h"

namespace luminous {
    inline namespace render {
        struct FilterSample {
            float2 p;
            float weight;
        };

        using lstd::Variant;
        class FilterHandle : public Variant<BoxFilter, GaussianFilter, TriangleFilter> {
            using Variant::Variant;
        public:
            CPU_ONLY(_NODISCARD std::string to_string() const;)

            NDSC_XPU float2 radius() const;

            NDSC_XPU float integral() const;

            NDSC_XPU float evaluate(float2 p) const;

            NDSC_XPU FilterSample sample(float2 u) const;

            static FilterHandle create(const FilterConfig &config);
        }
    }
}