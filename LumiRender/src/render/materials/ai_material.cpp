//
// Created by Zero on 2021/5/13.
//

#include "render/scene/scene_data.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDF AssimpMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            Texture tex = scene_data->textures[Kd_idx];
            BxDF bxdf = BxDF(IdealDiffuse(tex.eval(ctx)));
            return BSDF(ctx.ng, ctx.ns, ctx.dp_dus, bxdf);
        }
    }
}