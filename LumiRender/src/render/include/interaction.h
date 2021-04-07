//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        struct Interaction {
            float3 pos;
            float3 ng;
            float3 ns;
            float2 uv;
            float3 wo;
            float time;
        };
    }
}