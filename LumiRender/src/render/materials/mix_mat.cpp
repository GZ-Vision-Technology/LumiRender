//
// Created by Zero on 17/03/2022.
//

#include "mix_mat.h"

namespace luminous {
    inline namespace render{
        BSDFWrapper MixMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            return {};
        }
    }
}