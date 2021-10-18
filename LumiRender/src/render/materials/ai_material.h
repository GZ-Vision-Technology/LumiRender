//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/textures/texture.h"
#include "render/include/config.h"
#include "render/bxdfs/bsdf.h"
#include "core/refl/reflection.h"

namespace luminous {
    inline namespace render {
        class AssimpMaterial : BASE_CLASS() {
        public:
            REFL_CLASS(AssimpMaterial)
        private:
            index_t Kd_idx{};
            index_t Ks_idx{};
            index_t _normal_idx{};

            float4 Kd{};
            float4 Ks{};
        public:
            CPU_ONLY(explicit AssimpMaterial(const MaterialConfig &mc)
                             :AssimpMaterial(mc.diffuse_tex.tex_idx, mc.specular_tex.tex_idx,
                             mc.normal_tex.tex_idx, mc.diffuse_tex.val,
                             mc.specular_tex.val) {})

            AssimpMaterial(index_t diffuse_idx, index_t specular_idx,
                           index_t normal_idx, float4 diffuse, float4 specular)
                    : Kd_idx(diffuse_idx),
                      Ks_idx(specular_idx),
                      _normal_idx(normal_idx),
                      Kd(diffuse),
                      Ks(specular) {}

            LM_ND_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;
        };
    }
}