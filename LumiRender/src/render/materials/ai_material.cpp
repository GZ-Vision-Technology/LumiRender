//
// Created by Zero on 2021/5/13.
//

#include "render/scene/scene_data.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper AssimpMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float4 color = scene_data->get_texture(_color_idx).eval(ctx);
//            DiffuseBSDF diffuse_bsdf = create_diffuse_bsdf(color);
//            BSDF bsdf{diffuse_bsdf};
            DisneyBSDFData data;
            data.color = color;
            BSDF bsdf{data.create()};
            return {ctx.ng, ctx.ns, ctx.dp_dus, bsdf};
        }
    }
}