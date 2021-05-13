//
// Created by Zero on 2021/5/13.
//

#include "render/include/shader_data.h"
#include "ai_material.h"

namespace luminous {
    inline namespace render {

        BSDF AssimpMaterial::get_BSDF(const MaterialEvalContext &ctx, const HitGroupData *hit_group_data) const {
            Texture tex = hit_group_data->textures[_Kd_idx];
            BxDF bxdf = BxDF(IdealDiffuse(tex.eval(ctx)));
            return BSDF(ctx.ng, ctx.ns, ctx.dp_dus, bxdf);
        }
    }
}