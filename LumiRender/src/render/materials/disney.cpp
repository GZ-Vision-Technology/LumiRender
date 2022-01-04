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
            // todo merge texture
            float4 color = scene_data->get_texture(_color).eval(ctx);
            float metallic = scene_data->get_texture(_metallic).eval(ctx).x;
            float eta = scene_data->get_texture(_eta).eval(ctx).x;
            float spec_trans = scene_data->get_texture(_spec_trans).eval(ctx).x;
            float diffuse_weight = (1 - metallic) * (1 - spec_trans);
            float diff_trans = scene_data->get_texture(_diff_trans).eval(ctx).x / 2.f;
            float spec_tint = scene_data->get_texture(_specular_tint).eval(ctx).x;
            float roughness = scene_data->get_texture(_roughness).eval(ctx).x;
            float lum = Spectrum{color}.Y();
            Spectrum color_tint = lum > 0 ? (color / lum) : Spectrum(1.f);
            float sheen_weight = scene_data->get_texture(_sheen).eval(ctx).x;
            float sheen_tint = scene_data->get_texture(_sheen_tint).eval(ctx).x;
            Spectrum color_sheen_tint = sheen_weight > 0.f ?
                                      lerp(sheen_tint, Spectrum{1.f}, color_tint) :
                                      Spectrum(0.f);
            float4 scatter_distance = scene_data->get_texture(_scatter_distance).eval(ctx);

            DisneyBSDF disney_bsdf;

            if (_thin) {
                float flatness = scene_data->get_texture(_flatness).eval(ctx).x;
                disney::Diffuse diffuse(diffuse_weight * (1 - flatness) * (1 - diff_trans));
                disney_bsdf.add_BxDF(diffuse);
                disney::FakeSS fake_ss(diffuse_weight * flatness * (1 - diff_trans));
                disney_bsdf.add_BxDF(fake_ss);
            } else {

                if (is_zero(scatter_distance)) {
                    disney_bsdf.add_BxDF(disney::Diffuse(diffuse_weight));
                } else {
                    disney_bsdf.add_BxDF(disney::SpecularTransmission());
                    // todo process BSSRDF
                }
            }

            disney_bsdf.add_BxDF(disney::Retro(diffuse_weight));

            disney_bsdf.add_BxDF(disney::Sheen(diffuse_weight * sheen_weight));

            float aspect = safe_sqrt(1 - scene_data->get_texture(_anisotropic).eval(ctx).x * 0.9f);
            float ax = std::max(0.001f, sqr(roughness) / aspect);
            float ay = std::max(0.001f, sqr(roughness) * aspect);
            Spectrum R0 = lerp(metallic,
                               schlick_R0_from_eta(eta) * lerp(spec_tint, Spectrum{1.f}, color_sheen_tint),
                               Spectrum(color));

            return {ctx.ng, ctx.ns, ctx.dp_dus, BSDF{disney_bsdf}};
        }
    }
}