//
// Created by Zero on 17/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "render/textures/attr.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class GlassMaterial {

            DECLARE_REFLECTION(GlassMaterial)

        private:
            Attr3D _color{};
            Attr1D _eta{};
            Attr2D _roughness{};
            bool _remapping_roughness{};
        public:
            explicit GlassMaterial(Attr3D color, Attr1D eta, Attr2D roughness, bool remapping_roughness)
                    : _color(color), _eta(eta),
                      _roughness(roughness),
                      _remapping_roughness(remapping_roughness) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit GlassMaterial(const MaterialConfig &mc)
                    : GlassMaterial(Attr3D(mc.color),
                                    Attr1D(mc.eta),
                                    Attr2D(mc.roughness),
                                    mc.remapping_roughness) {})
        };
    }
}