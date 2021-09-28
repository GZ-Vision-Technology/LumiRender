//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/bxdfs/bsdf.h"
#include "render/include/config.h"
#include "core/concepts.h"


namespace luminous {
    inline namespace render {
        class MatteMaterial : BASE_CLASS(), public Creator<MatteMaterial> {
        public:
            REFL_CLASS(MatteMaterial)
        private:
            index_t _R{};
        public:
            explicit MatteMaterial(index_t r) : _R(r) {}

            NDSC_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MatteMaterial(const MaterialConfig &mc)
                             :MatteMaterial(mc.diffuse_tex.tex_idx) {})
        };
    }
}