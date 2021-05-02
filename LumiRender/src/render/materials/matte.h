//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/bxdfs/bsdf.h"

namespace luminous {
    inline namespace render {
        class MatteMaterial {
        private:
            index_t _R;
        public:
            NDSC_XPU BSDF get_BSDF(TextureEvalContext tec, const HitGroupData* hit_group_data, BxDF *bxdf) {

            }
        };
    }
}