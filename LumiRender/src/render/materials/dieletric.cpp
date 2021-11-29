//
// Created by Zero on 2021/6/9.
//

#include "dieletric.h"

namespace luminous {
    inline namespace render {

        BSDF DielectricMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            return BSDF();
        }
    }
}