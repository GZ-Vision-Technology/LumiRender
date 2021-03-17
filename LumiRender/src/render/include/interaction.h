//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        struct Interaction {
            float3 position;
            float3 normal;
            float3 wo;
            float time;
        };
    }
}