//
// Created by Zero on 18/12/2021.
//

#include "metal.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper FakeMetalMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float4 color = scene_data->get_texture(_color_idx).eval(ctx);
            const Texture &roughness_tex = scene_data->get_texture(_roughness_idx);
            float2 roughness = make_float2(roughness_tex.eval(ctx));
            if (_remapping_roughness) {
                roughness.x = MicrofacetDistrib::roughness_to_alpha(roughness.x);
                roughness.y = MicrofacetDistrib::roughness_to_alpha(roughness.y);
            }
            static constexpr auto min_roughness = 0.001f;
            // todo change to vector compute
            roughness.x = roughness.x < min_roughness ? min_roughness : roughness.x;
            roughness.y = roughness.y < min_roughness ? min_roughness : roughness.y;
            FakeMetalBSDF fake_metal_material = create_fake_metal_bsdf(color, roughness.x, roughness.y);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{fake_metal_material}};
        }


        BSDFWrapper MetalMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            //todo frequently look up the texture lead to time consuming to increase
            float4 eta = scene_data->get_texture(_eta_idx).eval(ctx);
            float2 roughness = make_float2(scene_data->get_texture(_roughness_idx).eval(ctx));
            if (_remapping_roughness) {
                roughness.x = MicrofacetDistrib::roughness_to_alpha(roughness.x);
                roughness.y = MicrofacetDistrib::roughness_to_alpha(roughness.y);
            }
            static constexpr auto min_roughness = 0.001f;
            roughness.x = roughness.x < min_roughness ? min_roughness : roughness.x;
            roughness.y = roughness.y < min_roughness ? min_roughness : roughness.y;
            float4 k = scene_data->get_texture(_k_idx).eval(ctx);
            MetalBSDF metal_bsdf = create_metal_bsdf(eta, k, roughness.x, roughness.y);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{metal_bsdf}};
        }
    }
}