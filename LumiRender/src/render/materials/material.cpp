//
// Created by Zero on 26/09/2021.
//

#include "material.h"
#include "common.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper Material::get_BSDF(MaterialEvalContext ctx, const SceneData *scene_data) const {
            LUMINOUS_VAR_PTR_DISPATCH(get_BSDF, ctx, scene_data)
        }

        void Material::compute_shading_frame(MaterialEvalContext *ctx, const SceneData *scene_data) const {
            float3 normal = make_float3(scene_data->get_texture(_normal_idx).eval(*ctx)) * 2.f - make_float3(1.f);
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