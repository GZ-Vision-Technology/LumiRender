//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "attr.h"
#include "core/concepts.h"


namespace luminous {
    inline namespace render {
        class MatteMaterial {
        public:
            DECLARE_REFLECTION(MatteMaterial)
        private:
            index_t _color_idx{};
            float _sigma{};
        public:
            explicit MatteMaterial(index_t r, float sigma) : _color_idx(r), _sigma(sigma) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MatteMaterial(const MaterialConfig &mc)
                             :MatteMaterial(mc.color_tex.tex_idx(),
                                            mc.sigma) {})
        };
    }
}