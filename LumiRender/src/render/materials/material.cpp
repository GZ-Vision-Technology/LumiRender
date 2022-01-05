//
// Created by Zero on 26/09/2021.
//

#include "material.h"
#include "common.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper Material::get_BSDF(MaterialEvalContext ctx, const SceneData *scene_data) const {

          LUMINOUS_VAR_PTR_DISPATCH(get_BSDF, ctx, scene_data)
        }

#ifndef __CUDACC__
        std::pair<Material, std::vector<size_t>> Material::create(const MaterialConfig &mc) {
            auto ret = detail::create_ptr<Material>(mc);
            ret.first._normal_idx = mc.normal_tex.tex_idx();
            return ret;
        }
#endif
    }
}