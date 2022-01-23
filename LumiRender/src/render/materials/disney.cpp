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
            data.color = make_float4(_color.eval(scene_data, ctx), 1.f);
            data.metallic = _metallic.eval(scene_data, ctx);
            data.eta = _eta.eval(scene_data, ctx);
            data.spec_trans = _spec_trans.eval(scene_data, ctx);
            data.diff_trans = _diff_trans.eval(scene_data, ctx);
            data.spec_tint = _specular_tint.eval(scene_data, ctx);
            data.roughness = _roughness.eval(scene_data, ctx);
            data.sheen_weight = _sheen.eval(scene_data, ctx);
            data.sheen_tint = _sheen_tint.eval(scene_data, ctx);
            data.clearcoat = _clearcoat.eval(scene_data, ctx);
            data.scatter_distance = make_float4(_scatter_distance.eval(scene_data, ctx), 0.f);
            data.clearcoat_roughness = _clearcoat_roughness.eval(scene_data, ctx);
            data.anisotropic = _anisotropic.eval(scene_data, ctx);
            data.flatness = _flatness.eval(scene_data, ctx);
            auto disney_bsdf = data.create();
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{disney_bsdf}};
        }
    }
}