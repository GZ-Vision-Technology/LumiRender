//
// Created by Zero on 2021/7/28.
//


#pragma once

#include <utility>

#include "light_base.h"
#include "render/distribution/distribution.h"
#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {
        class Scene;
        class Envmap : public LightBase {
        private:
            Distribution2D _distribution;
            index_t _tex_idx{};
            Transform _o2w;
            float3 _scene_center{};
            float _scene_radius{};
        public:
            Envmap(index_t tex_idx, Transform o2w, Distribution2D distribution)
                : LightBase(LightType::Infinite),
                _tex_idx(tex_idx),
                _o2w(o2w),
                _distribution(distribution) {}

            void preprocess(const Scene *scene);

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU SurfaceInteraction sample(float2 u, const HitGroupData *hit_group_data) const;

            NDSC_XPU float PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light) const;

            NDSC_XPU Spectrum power() const;

            XPU void print() const;

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("light Base : %s,name:%s",
                                   LightBase::to_string().c_str(),
                                   type_name(this));
            })

            CPU_ONLY(static std::vector<float> create_distribution(const Image &image));

            CPU_ONLY(static Envmap create(const LightConfig &config);)
        };
    }
}