//
// Created by Zero on 2021/5/13.
//

#include "render/scene/scene_data.h"
#include "matte.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper MatteMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            const Texture& tex = scene_data->get_texture(_color_idx);
            float4 color = tex.eval(ctx);
            if (is_valid_index(_sigma)) {
                const Texture& sigma_tex = scene_data->get_texture(_sigma);
                float sigma = sigma_tex.eval(ctx).x;
                OrenNayarBSDF oren_nayar_bsdf = create_oren_nayar_bsdf(color, sigma);
                BSDF bsdf{oren_nayar_bsdf};
                return {ctx.ng, ctx.ns, ctx.dp_dus, bsdf};
            }
            DiffuseBSDF diffuse_bsdf = create_diffuse_bsdf(color);
            BSDF bsdf{diffuse_bsdf};
            return {ctx.ng, ctx.ns, ctx.dp_dus, bsdf};
        }
    }
}