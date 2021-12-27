//
// Created by Zero on 18/12/2021.
//

#include "metal.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper FakeMetalMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            const Texture& tex = scene_data->get_texture(_color_idx);
            float4 color = tex.eval(ctx);
            MirrorBSDF mirror_bsdf = create_mirror_bsdf(color);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{mirror_bsdf}};
        }
    }
}