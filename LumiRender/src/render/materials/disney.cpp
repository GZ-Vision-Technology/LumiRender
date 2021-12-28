//
// Created by Zero on 28/12/2021.
//

#include "disney.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper DisneyMaterial::get_BSDF(const MaterialEvalContext &ctx,
                                             const SceneData *scene_data) const {
            return BSDFWrapper();
        }
    }
}