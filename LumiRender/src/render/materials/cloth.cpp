#include "cloth.h"
#include "parser/config.h"
#include "render/scattering/bsdfs.h"

namespace luminous {
    inline namespace render {

#ifndef __NVCC__
        ClothMaterial::ClothMaterial(const MaterialConfig &config)
            : ClothMaterial(Attr3D(config.color), Attr1D(config.eta), Attr1D(config.roughness), config.remapping_roughness) {
        }
#endif

    BSDFWrapper ClothMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene) const {

        Spectrum base_color = _base_color.eval(scene, ctx);
        float2 roughness = float2{ _roughness.eval(scene, ctx) };
        if(_remapping_roughness) {
            roughness.x = Microfacet::roughness_to_alpha(roughness.x);
            roughness.y = roughness.x;
        }

        static constexpr float2 min_roughness = float2{ 0.001f };
        roughness = min(roughness, min_roughness);

        float eta = _eta.eval(scene, ctx);

        ClothMicrofacetFresnel bxdf{ base_color, Spectrum{0.0f}, roughness.x,
            &scene->cloth_spec_albedos[0], &scene->cloth_spec_albedos[1] };

        BSDFHelper data;
        Spectrum R0 = schlick_R0_from_eta(eta);
        data.set_R0(R0);
        data.set_eta(eta);

        return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{ NeubeltClothBSDF{data, bxdf} }};
    }

    }// namespace render
}// namespace luminous