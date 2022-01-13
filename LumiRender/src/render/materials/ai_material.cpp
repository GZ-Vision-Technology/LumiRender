//
// Created by Zero on 2021/5/13.
//

#include "render/scene/scene_data.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper AssimpMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float4 color = scene_data->get_texture(_color_idx).eval(ctx);
            DisneyBSDFData data;
            data.color = color;
            float4 spec = scene_data->get_texture(_specular_idx).eval(ctx);
            data.roughness = 1 - spec.x;
            data.clearcoat = spec.x;
            data.clearcoat_gloss = spec.x;
            data.sheen_weight = spec.x;
            data.sheen_tint = spec.x;
            data.spec_tint = spec.x;
            data.metallic = spec.x;
            BSDF bsdf{data.create()};
            return {ctx.ng, ctx.ns, ctx.dp_dus, bsdf};
        }
    }
}