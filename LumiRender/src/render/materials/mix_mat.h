//
// Created by Zero on 17/03/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "render/textures/attr.h"
#include "core/concepts.h"
#include "material.h"

namespace luminous {
    inline namespace render {
        class MixMaterial {
        DECLARE_REFLECTION(MixMaterial)

        private:
            Material _material[2];
            Attr1D _scale;

        public:
            MixMaterial(Material mat0, Material mat1, Attr1D scale)
                    : _scale(scale) {
                _material[0] = mat0;
                _material[1] = mat1;
            }

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

        };
    }
}