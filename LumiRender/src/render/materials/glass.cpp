//
// Created by Zero on 2021/6/9.
//

#include "glass.h"

namespace luminous {
    inline namespace render {

        BSDF GlassMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            return BSDF();
        }
    }
}