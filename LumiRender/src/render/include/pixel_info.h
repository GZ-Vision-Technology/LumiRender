//
// Created by Zero on 12/10/2021.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace render {
        struct PixelInfo {
        public:
            Spectrum Li{};
            float3 normal{};
            float3 albedo{};
        };
    }
}