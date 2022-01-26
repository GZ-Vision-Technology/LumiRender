//
// Created by Zero on 2021/7/26.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/type_reflection.h"
#include "filter_sampler.h"

namespace luminous {
    inline namespace render {

        class FilterBase {

            DECLARE_REFLECTION(FilterBase)

        protected:
            const float2 _radius;
        public:
            explicit FilterBase(const float2 r)
                    : _radius(r) {}

            LM_ND_XPU float2 radius() const {
                return _radius;
            }

            GEN_STRING_FUNC({
                                return string_printf("filter radius:%s", _radius.to_string().c_str());
                            })
        };

        class FittedFilter : public FilterBase {

            DECLARE_REFLECTION(FittedFilter, FilterBase)

        protected:
            FilterSampler _sampler;
        public:
            explicit FittedFilter(const float2 r)
                    : FilterBase(r) {}

            LM_ND_XPU FilterSample sample(const float2 &u) const {
                FilterSample fs = _sampler.sample(u);
                fs.p = fs.p * radius();
                return fs;
            }
        };
    }
}