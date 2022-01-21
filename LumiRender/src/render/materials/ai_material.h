//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/textures/texture.h"
#include "parser/config.h"
#include "render/scattering/bsdf_wrapper.h"
#include "core/type_reflection.h"

namespace luminous {
    inline namespace render {
        class AssimpMaterial {
        public:
            DECLARE_REFLECTION(AssimpMaterial)

        private:
            index_t _color_idx{};
            index_t _specular_idx{};
            index_t _normal_idx{};

            float4 _color{};
            float4 _specular{};
        public:
            CPU_ONLY(explicit AssimpMaterial(const MaterialConfig &mc)
                    : AssimpMaterial(mc.color_tex.tex_idx(), mc.specular_tex.tex_idx(),
                                     mc.normal_tex.tex_idx(), mc.color_tex.val,
                                     mc.specular_tex.val) {})

            AssimpMaterial(index_t diffuse_idx, index_t specular_idx,
                           index_t normal_idx, float4 diffuse, float4 specular)
                    : _color_idx(diffuse_idx),
                      _specular_idx(specular_idx),
                      _normal_idx(normal_idx),
                      _color(diffuse),
                      _specular(specular) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;
        };
    }
}