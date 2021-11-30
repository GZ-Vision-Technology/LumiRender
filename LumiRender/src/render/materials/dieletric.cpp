//
// Created by Zero on 2021/6/9.
//

#include "dieletric.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDF DielectricMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            const Texture &kr_tex = scene_data->get_texture(Kr);
            const Texture &kt_tex = scene_data->get_texture(Kt);
            const Texture &roughness_tex = scene_data->get_texture(_roughness);
            const Texture &eta_tex = scene_data->get_texture(_roughness);

            float4 roughness = roughness_tex.eval(ctx);
            float rx{0}, ry{0};
            if (roughness_tex.channel_num() == 1) {
                rx = ry = roughness.x;
            } else {
                rx = roughness.x;
                ry = roughness.y;
            }
            MicrofacetDistribution distribution(rx, ry, GGX);
            float4 kr = kr_tex.eval(ctx);
            float4 kt = kt_tex.eval(ctx);
            float eta = eta_tex.eval(ctx).x;

            BxDF bxdf = BxDF(DielectricBxDF(kr, kt, eta, distribution));

            return {ctx.ng, ctx.ns, ctx.dp_dus, bxdf};
        }
    }
}