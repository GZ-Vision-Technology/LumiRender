//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/bxdfs/bsdf.h"
#include "render/include/config.h"


namespace luminous {
    inline namespace render {
        class MatteMaterial {
        private:
            index_t _R;
        public:
            MatteMaterial(index_t r) : _R(r) {}

            NDSC_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const HitGroupData* hit_group_data) const;

            CPU_ONLY(static MatteMaterial create(const MaterialConfig &mc) {
                return MatteMaterial(mc.diffuse_idx);
            })
        };
    }
}