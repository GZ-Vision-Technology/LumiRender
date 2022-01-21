//
// Created by Zero on 2021/7/28.
//


#pragma once

#include "light_base.h"
#include "base_libs/sampling/distribution.h"
#include "render/textures/texture.h"
#ifndef __CUDACC__
#include "util/image.h"
#endif

namespace luminous {
    inline namespace render {
        class Scene;

        // todo make CPU support
        class Envmap : public LightBase {
        public:
            DECLARE_REFLECTION(Envmap, LightBase)
        private:
            index_t _tex_idx{invalid_uint32};
            index_t _distribution_idx{invalid_uint32};
            Transform _w2o;
            float3 _scene_center{};
            float _scene_diameter{};
        public:
            CPU_ONLY(explicit Envmap(const LightConfig &config)
                    : LightBase(LightType::Infinite),
                      _tex_idx(config.texture_config.tex_idx()),
                      _scene_center(config.scene_box.center()),
                      _scene_diameter(config.scene_box.radius() * 2.f),
                      _distribution_idx(config.distribution_idx) {
                Transform o2w = config.o2w_config.create();
                Transform rotate_x = Transform::rotation_x(90);
                _w2o = (o2w * rotate_x).inverse();
            })

            Envmap(index_t tex_idx, Transform w2o, index_t distribution_idx, Box3f scene_box)
                    : LightBase(LightType::Infinite),
                      _tex_idx(tex_idx),
                      _w2o(w2o),
                      _scene_center(scene_box.center()),
                      _scene_diameter(scene_box.radius() * 2.f),
                      _distribution_idx(distribution_idx) {}

            LM_ND_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            LM_ND_XPU Spectrum L(float3 dir_in_obj, const SceneData *data) const;

            LM_ND_XPU LightEvalContext sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            LM_ND_XPU float PDF_Li(const LightSampleContext &p_ref, const LightEvalContext &p_light,
                                   float3 wi, const SceneData *data) const;

            LM_ND_XPU Spectrum on_miss(float3 dir, const SceneData *data) const;

            LM_ND_XPU Spectrum power() const;

            LM_XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s,name:%s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this));
                            })

            CPU_ONLY(static std::vector<float> create_distribution(const Image &image));
        };
    }
}