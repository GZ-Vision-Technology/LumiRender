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
        class MirrorMaterial {

            DECLARE_REFLECTION(MirrorMaterial)
        private:
            Attr3D _color{};
        public:
            explicit MirrorMaterial(Attr3D color) : _color(color) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MirrorMaterial(const MaterialConfig &mc)
                    : MirrorMaterial(Attr3D(mc.color)) {})
        };
    }
}