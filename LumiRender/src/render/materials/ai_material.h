//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/textures/texture.h"
#include "attr.h"
#include "render/scattering/bsdf_wrapper.h"
#include "core/type_reflection.h"

namespace luminous {
    inline namespace render {
        class AssimpMaterial {
        public:
            DECLARE_REFLECTION(AssimpMaterial)

        private:
            Attr3D _color{};
            Attr3D _specular{};

        public:
            CPU_ONLY(explicit AssimpMaterial(const MaterialConfig &mc)
                    : AssimpMaterial(Attr3D(mc.color),
                                     Attr3D(mc.specular)) {})

            AssimpMaterial(Attr3D diffuse, Attr3D specular)
                    : _color(diffuse),
                      _specular(specular) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;
        };
    }
}