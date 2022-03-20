//
// Created by Zero on 22/02/2022.
//

#include "substrate.h"

namespace luminous {
    inline namespace render {
        BSDFWrapper SubstrateMaterial::get_BSDF(const MaterialEvalContext &ctx,
                                                const SceneData *scene_data) const {
            float3 color = _color.eval(scene_data, ctx);
            float3 spec = _spec.eval(scene_data, ctx);
            float2 roughness = _roughness.eval(scene_data, ctx);
            if (_remapping_roughness) {
                roughness.x = Microfacet::roughness_to_alpha(roughness.x);
                roughness.y = Microfacet::roughness_to_alpha(roughness.y);
            }
            static constexpr auto min_roughness = make_float2(0.001f);
            roughness = select(roughness < min_roughness, min_roughness, roughness);
            SubstrateBSDF bsdf = create_substrate_bsdf(color, spec, roughness);
            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF(bsdf)};
        }
    }
}