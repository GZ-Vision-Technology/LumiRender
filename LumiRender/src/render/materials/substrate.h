//
// Created by Zero on 22/02/2022.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "render/textures/attr.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class SubstrateMaterial {
        DECLARE_REFLECTION(SubstrateMaterial)

        private:
            Attr3D _color{};
            Attr3D _spec{};
            Attr2D _roughness{};
            bool _remapping_roughness{};
        public:
            explicit SubstrateMaterial(Attr3D color, Attr3D spec, Attr2D roughness, bool remapping_roughness)
                    : _color(color), _spec(spec),
                      _roughness(roughness),
                      _remapping_roughness(remapping_roughness) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit SubstrateMaterial(const MaterialConfig &mc)
                    : SubstrateMaterial(Attr3D(mc.color),
                                        Attr3D(mc.specular),
                                        Attr2D(mc.roughness),
                                        mc.remapping_roughness) {})
        };
    }
}