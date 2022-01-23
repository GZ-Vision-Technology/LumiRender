//
// Created by Zero on 2021/5/13.
//

#include "render/scene/scene_data.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper AssimpMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float4 color = make_float4(_color.eval(scene_data, ctx), 1.f);
            DisneyBSDFData data;
            data.color = color;
            float4 spec = make_float4(_specular.eval(scene_data, ctx), 1.f);
            data.roughness = 1 - spec.x;
            data.clearcoat = spec.x;
            data.clearcoat_roughness = 1 - spec.x;
            data.sheen_weight = spec.x;
            data.sheen_tint = spec.x;
            data.spec_tint = spec.x;
            data.metallic = spec.x;
            BSDF bsdf{data.create()};
            return {ctx.ng, ctx.ns, ctx.dp_dus, bsdf};
        }
    }
}