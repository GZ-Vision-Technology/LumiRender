//
// Created by Zero on 28/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "attr.h"
#include "core/concepts.h"
#include "core/type_reflection.h"

namespace luminous {
    inline namespace render {
        class DisneyMaterial {
        public:
            DECLARE_REFLECTION(DisneyMaterial)

        private:
            Attr3D _color{};
            Attr1D _metallic{};
            Attr1D _eta{};
            Attr1D _roughness{};
            Attr1D _specular_tint{};
            Attr1D _anisotropic{};
            Attr1D _sheen{};
            Attr1D _sheen_tint{};
            Attr1D _clearcoat{};
            Attr1D _clearcoat_roughness{};
            Attr1D _spec_trans{};
            Attr3D _scatter_distance{};
            Attr1D _flatness{};
            Attr1D _diff_trans{};
            bool _thin{};

        public:

            LM_XPU DisneyMaterial(Attr3D color,
                                  Attr1D metallic,
                                  Attr1D eta,
                                  Attr1D roughness,
                                  Attr1D specular_tint,
                                  Attr1D anisotropic,
                                  Attr1D sheen,
                                  Attr1D sheen_tint,
                                  Attr1D clearcoat,
                                  Attr1D clearcoat_roughness,
                                  Attr1D spec_trans,
                                  Attr3D scatter_distance,
                                  bool thin,
                                  Attr1D flatness,
                                  Attr1D diff_trans)
                    : _color(color),
                      _metallic(metallic),
                      _eta(eta),
                      _roughness(roughness),
                      _specular_tint(specular_tint),
                      _anisotropic(anisotropic),
                      _sheen(sheen),
                      _sheen_tint(sheen_tint),
                      _clearcoat(clearcoat),
                      _clearcoat_roughness(clearcoat_roughness),
                      _spec_trans(spec_trans),
                      _scatter_distance(scatter_distance),
                      _thin(thin),
                      _flatness(flatness),
                      _diff_trans(diff_trans) {}

            CPU_ONLY(explicit DisneyMaterial(const MaterialConfig &mc)
                    : DisneyMaterial(Attr3D(mc.color),
                                     Attr1D(mc.metallic),
                                     Attr1D(mc.eta),
                                     Attr1D(mc.roughness),
                                     Attr1D(mc.specular_tint),
                                     Attr1D(mc.anisotropic),
                                     Attr1D(mc.sheen),
                                     Attr1D(mc.sheen_tint),
                                     Attr1D(mc.clearcoat),
                                     Attr1D(mc.clearcoat_roughness),
                                     Attr1D(mc.spec_trans),
                                     Attr3D(mc.scatter_distance),
                                     mc.thin,
                                     Attr1D(mc.flatness),
                                     Attr1D(mc.diff_trans)) {})

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx,
                                           const SceneData *scene_data) const;
        };
    }
}