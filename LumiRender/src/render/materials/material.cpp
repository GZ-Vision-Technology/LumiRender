//
// Created by Zero on 26/09/2021.
//

#include "material.h"
#include "common.h"

namespace luminous {
    inline namespace render {

        BSDF Material::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {

          LUMINOUS_VAR_PTR_DISPATCH(get_BSDF, ctx, scene_data)
        }

#ifndef __CUDACC__
        std::pair<Material, std::vector<size_t>> Material::create(const MaterialConfig &mc) {
            return detail::create_ptr<Material>(mc);
        }
#endif
    }
}