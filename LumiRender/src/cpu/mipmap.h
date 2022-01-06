//
// Created by Zero on 06/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "util/image.h"
#include "core/memory/block_array.h"

namespace luminous {
    inline namespace cpu {

        struct ResampleWeight {
            int firstTexel;
            float weight[4];
        };

        class MIPMap {
        private:
            const bool _tri_linear;

            const float _max_anisotropy;

            const ImageWrap _wrap_mode;

            uint2 _resolution;

            std::vector<std::unique_ptr<BlockedArray<float>>> _pyramid;
        public:
        };
    }
}