//
// Created by Zero on 06/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "util/image.h"
#include "pyramid_mgr.h"

namespace luminous {
    inline namespace cpu {

        struct ResampleWeight {
            int firstTexel;
            float weight[4];
        };

        class MIPMap : public ImageBase {
        private:
            const bool _tri_linear{true};
            const float _max_anisotropy{8.f};
            index_t _index{invalid_uint32};

        public:
            MIPMap(const Image &image, float max_anisotropy, bool tri_linear)
                    : ImageBase(image.pixel_format(), image.resolution()) {}

            void gen_pyramid();

            void resample_weight();
        };
    }
}