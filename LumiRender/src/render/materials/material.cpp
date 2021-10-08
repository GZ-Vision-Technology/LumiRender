//
// Created by Zero on 26/09/2021.
//

#include "material.h"
#include "matte.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDF Material::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            LUMINOUS_VAR_PTR_DISPATCH(get_BSDF, ctx, scene_data)
        }

        CPU_ONLY(Material Material::create(const MaterialConfig &mc) {
            return detail::create_ptr<Material>(mc);
        })
    }
}