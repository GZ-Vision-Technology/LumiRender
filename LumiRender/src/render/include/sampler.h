//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {

        class SamplerBase {
        protected:
            int _spp;
        public:
            [[nodiscard]] int spp() const { return _spp; }
        };

        class LCGSampler;
        class PCGSampler;

        using lstd::Variant;
        class SamplerHandler : public Variant<LCGSampler*, PCGSampler*> {

        };

    }
}