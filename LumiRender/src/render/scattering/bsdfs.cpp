//
// Created by Zero on 13/01/2022.
//

#include "bsdfs.h"

namespace luminous {
    inline namespace render {

        DisneyBSDF DisneyBSDFData::create() const {
            float diffuse_weight = (1 - metallic) * (1 - spec_trans);
            float lum = Spectrum{color}.Y();
            Spectrum color_tint = lum > 0 ? (color / lum) : Spectrum(1.f);
            Spectrum color_sheen_tint = sheen_weight > 0.f ?
                    lerp(sheen_tint, Spectrum{1.f}, color_tint) :
                    Spectrum(0.f);
            Spectrum R0 = lerp(metallic,
                               schlick_R0_from_eta(eta) * lerp(spec_tint, Spectrum{1.f}, color_sheen_tint),
                               Spectrum(color));
            DisneyBSDF disney_bsdf;
            BSDFHelper helper{DisneyFr};
            helper.set_roughness(roughness);
            helper.set_metallic(metallic);
            helper.set_R0(R0);
            helper.set_eta(eta);
            helper.set_clearcoat_alpha(lerp(clearcoat_roughness, 0.001f, 0.1f));
            float dt = diff_trans / 2.f;
            disney_bsdf.set_data(helper);
            float aspect = safe_sqrt(1 - anisotropic * 0.9f);

            if (thin) {
                disney::Diffuse diffuse(diffuse_weight * (1 - flatness) * (1 - dt) * color);
                disney_bsdf.add_BxDF(diffuse);
                disney::FakeSS fake_ss(diffuse_weight * flatness * (1 - dt) * color);
                disney_bsdf.add_BxDF(fake_ss);
            } else {
                if (is_zero(scatter_distance)) {
                    disney_bsdf.add_BxDF(disney::Diffuse(diffuse_weight * color));
                } else {
                    disney_bsdf.add_BxDF(SpecularTransmission(Spectrum{1.f}));
                    // todo process BSSRDF
                }
            }

            disney_bsdf.add_BxDF(disney::Retro(diffuse_weight * color));

            disney_bsdf.add_BxDF(disney::Sheen(diffuse_weight * sheen_weight * color_sheen_tint));

            float ax = std::max(0.001f, sqr(roughness) / aspect);
            float ay = std::max(0.001f, sqr(roughness) * aspect);
            Microfacet distrib{ax, ay, MicrofacetType::Disney};
            disney_bsdf.add_BxDF(MicrofacetReflection(Spectrum{1.f}, distrib));
            disney_bsdf.add_BxDF(disney::Clearcoat{clearcoat});

            Spectrum T = spec_trans * sqrt(color);
            if (thin) {
                float rscaled = (0.65f * eta - 0.35f) * roughness;
                float ax = std::max(0.001f, sqr(rscaled) / aspect);
                float ay = std::max(0.001f, sqr(rscaled) * aspect);
                Microfacet distrib{ax, ay, GGX};
                disney_bsdf.add_BxDF(MicrofacetTransmission{T, distrib});
                disney_bsdf.add_BxDF(DiffuseTransmission(dt * color));

            } else {
                disney_bsdf.add_BxDF(MicrofacetTransmission{T, distrib});
            }
            return disney_bsdf;
        }
    }
}