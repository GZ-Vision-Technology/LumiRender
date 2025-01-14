//
// Created by Zero on 17/12/2021.
//

#include "mirror.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper MirrorMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float3 color = _color.eval(scene_data, ctx);
            MirrorBSDF mirror_bsdf = create_mirror_bsdf(color);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{mirror_bsdf}};
        }
    }
}