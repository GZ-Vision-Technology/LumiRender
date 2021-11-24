//
// Created by Zero on 2021/7/26.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace render {

        struct FilterSample {
            float2 p;
            float weight{};
        };

        struct FilterBase {
        protected:
            const float2 _radius;
        public:
            explicit FilterBase(const float2 r) : _radius(r) {}

            LM_ND_XPU float2 radius() const {
                return _radius;
            }

            GEN_STRING_FUNC({
                return string_printf("filter radius:%s", _radius.to_string().c_str());
            })
        };
    }
}