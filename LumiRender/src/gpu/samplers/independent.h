//
// Created by Zero on 2021/1/25.
//


#pragma once

#include "../owl/include/owl/common/math/random.h"
#include "core/math/data_types.h"

namespace luminous::sampler {
    class IndependentSampler {
    private:
        owl::common::LCG _lcg;
        uint _spp;

    public:
        IndependentSampler() {

        }

        XPU float next_1d() const {
            return _lcg();
        }
    };
}