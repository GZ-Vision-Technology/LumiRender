//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/textures/texture.h"
#include "render/include/config.h"
#include "render/bxdfs/bsdf.h"

namespace luminous {
    inline namespace render {
        class AssimpMaterial {
        private:
            index_t _Kd_idx;
            index_t _Ks_idx;
            index_t _normal_idx;

            float4 _Kd;
            float4 _Ks;
        public:
            AssimpMaterial(index_t diffuse_idx, index_t specular_idx,
                           index_t normal_idx, float4 diffuse, float4 specular)
                    : _Kd_idx(diffuse_idx),
                      _Ks_idx(specular_idx),
                      _normal_idx(normal_idx),
                      _Kd(diffuse),
                      _Ks(specular) {}

            NDSC_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const HitGroupData *hit_group_data) const;

            static AssimpMaterial create(const MaterialConfig &mc) {
                return AssimpMaterial(mc.diffuse_idx, mc.specular_idx, mc.normal_idx,
                                      mc.diffuse_tex.val, mc.specular_tex.val);
            }
        };
    }
}