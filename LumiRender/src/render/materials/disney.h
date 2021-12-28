//
// Created by Zero on 28/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "render/include/config.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class DisneyMaterial : BASE_CLASS() {
        public:
            REFL_CLASS(DisneyMaterial)

        private:
            index_t _color{};
            index_t _metallic{};
            index_t _eta{};
            index_t _roughness{};
            index_t _specular_tint{};
            index_t _anisotropic{};
            index_t _sheen{};
            index_t _sheen_tint{};
            index_t _clearcoat{};
            index_t _clearcoat_gloss{};
            index_t _spec_trans{};
            index_t _scatter_distance{};
            index_t _flatness{};
            index_t _diff_trans{};
            bool _thin{};

        public:

            LM_XPU DisneyMaterial(index_t color,
                                  index_t metallic,
                                  index_t eta,
                                  index_t roughness,
                                  index_t specular_tint,
                                  index_t anisotropic,
                                  index_t sheen,
                                  index_t sheen_tint,
                                  index_t clearcoat,
                                  index_t clearcoat_gloss,
                                  index_t spec_trans,
                                  index_t scatter_distance,
                                  bool thin,
                                  index_t flatness,
                                  index_t diff_trans)
                    : _color(color),
                      _metallic(metallic),
                      _eta(eta),
                      _roughness(roughness),
                      _specular_tint(specular_tint),
                      _anisotropic(anisotropic),
                      _sheen(sheen),
                      _sheen_tint(sheen_tint),
                      _clearcoat(clearcoat),
                      _clearcoat_gloss(clearcoat_gloss),
                      _spec_trans(spec_trans),
                      _scatter_distance(scatter_distance),
                      _thin(thin),
                      _flatness(flatness),
                      _diff_trans(diff_trans) {}

            CPU_ONLY(explicit DisneyMaterial(const MaterialConfig &mc)
                    : DisneyMaterial(mc.color_tex.tex_idx(),
                                     mc.metallic_tex.tex_idx(),
                                     mc.eta_tex.tex_idx(),
                                     mc.roughness_tex.tex_idx(),
                                     mc.specular_tint_tex.tex_idx(),
                                     mc.anisotropic_tex.tex_idx(),
                                     mc.sheen_tex.tex_idx(),
                                     mc.sheen_tint_tex.tex_idx(),
                                     mc.clearcoat_tex.tex_idx(),
                                     mc.clearcoat_gloss_tex.tex_idx(),
                                     mc.spec_trans_tex.tex_idx(),
                                     mc.scatter_distance_tex.tex_idx(),
                                     mc.thin,
                                     mc.flatness_tex.tex_idx(),
                                     mc.diff_trans_tex.tex_idx()) {})

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx,
                                           const SceneData *scene_data) const;
        };
    }
}