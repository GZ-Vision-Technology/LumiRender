//
// Created by Zero on 28/12/2021.
//

#include "disney.h"
#include "render/scene/scene_data.h"
#include "render/scattering/disney_bsdf.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper DisneyMaterial::get_BSDF(const MaterialEvalContext &ctx,
                                             const SceneData *scene_data) const {

            DisneyBSDFData data;

            // todo merge texture
            data.color = scene_data->get_texture(_color).eval(ctx);
            data.metallic = scene_data->get_texture(_metallic).eval(ctx).x;
            data.eta = scene_data->get_texture(_eta).eval(ctx).x;
            data.spec_trans = scene_data->get_texture(_spec_trans).eval(ctx).x;
            data.diff_trans = scene_data->get_texture(_diff_trans).eval(ctx).x / 2.f;
            data.spec_tint = scene_data->get_texture(_specular_tint).eval(ctx).x;
            data.roughness = scene_data->get_texture(_roughness).eval(ctx).x;
            data.sheen_weight = scene_data->get_texture(_sheen).eval(ctx).x;
            data.sheen_tint = scene_data->get_texture(_sheen_tint).eval(ctx).x;
            data.clearcoat = scene_data->get_texture(_clearcoat).eval(ctx).x;
            data.scatter_distance = scene_data->get_texture(_scatter_distance).eval(ctx);
            data.clearcoat_gloss = scene_data->get_texture(_clearcoat_gloss).eval(ctx).x;
            data.aspect = safe_sqrt(1 - scene_data->get_texture(_anisotropic).eval(ctx).x * 0.9f);
            data.flatness = scene_data->get_texture(_flatness).eval(ctx).x;

            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{data.create()}};
        }
    }
}