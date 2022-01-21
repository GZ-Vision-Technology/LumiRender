//
// Created by Zero on 18/12/2021.
//

#include "metal.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper FakeMetalMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
            float3 color = _color.eval(scene_data, ctx);
            float2 roughness = _roughness.eval(scene_data, ctx);
            if (_remapping_roughness) {
                roughness.x = Microfacet::roughness_to_alpha(roughness.x);
                roughness.y = Microfacet::roughness_to_alpha(roughness.y);
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
            float3 eta = _eta.eval(scene_data, ctx);
            float2 roughness = _roughness.eval(scene_data, ctx);
            if (_remapping_roughness) {
                roughness.x = Microfacet::roughness_to_alpha(roughness.x);
                roughness.y = Microfacet::roughness_to_alpha(roughness.y);
            }
            static constexpr auto min_roughness = 0.001f;
            roughness.x = roughness.x < min_roughness ? min_roughness : roughness.x;
            roughness.y = roughness.y < min_roughness ? min_roughness : roughness.y;
            float3 k = _k.eval(scene_data, ctx);
            MetalBSDF metal_bsdf = create_metal_bsdf(eta, k, roughness.x, roughness.y);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{metal_bsdf}};
        }
    }
}