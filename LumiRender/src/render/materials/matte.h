//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "render/textures/attr.h"
#include "core/concepts.h"


namespace luminous {
    inline namespace render {
        class MatteMaterial {

            DECLARE_REFLECTION(MatteMaterial)

        private:
            float _sigma{};

            Attr3D _color{};
        public:

            MatteMaterial(Attr3D color, float sigma) : _color(color), _sigma(sigma) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MatteMaterial(const MaterialConfig &mc)
                             :MatteMaterial(Attr3D(mc.color),
                                            mc.sigma) {})
        };
    }
}