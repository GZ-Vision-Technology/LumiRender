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
            index_t _tex_idx{index_t(-1)};
            index_t _distribution_idx{index_t(-1)};
            Transform _w2o;
            float3 _scene_center{};
            float _scene_radius{};
        public:
            Envmap(index_t tex_idx, Transform w2o, index_t distribution_idx, Box3f scene_box)
                    : LightBase(LightType::Infinite),
                      _tex_idx(tex_idx),
                      _w2o(w2o),
                      _scene_center(scene_box.center()),
                      _scene_radius(scene_box.radius()),
                      _distribution_idx(distribution_idx) {}

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU SurfaceInteraction sample(float2 u, const HitGroupData *hit_group_data) const;

            NDSC_XPU float PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light) const;

            NDSC_XPU Spectrum on_miss(Ray ray, const MissData *miss_data) const;

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