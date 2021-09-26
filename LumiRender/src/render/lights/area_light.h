//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "light_base.h"
#include "render/include/config.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class AreaLight : public LightBase, public ICreator<AreaLight> {
        public:
            uint _inst_idx{};
            float3 _L{};
            bool _two_sided{};
            float _inv_area{};
        public:
            CPU_ONLY(explicit AreaLight(const LightConfig &config)
                    : AreaLight(config.instance_idx, config.emission, config.surface_area,
                                config.two_sided) {})

            AreaLight(uint inst_idx, float3 L, float area, bool two_sided);

            NDSC_XPU Spectrum L(const SurfaceInteraction &p_light, float3 w) const;

            NDSC_XPU float inv_area() const;

            NDSC_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            NDSC_XPU SurfaceInteraction sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            NDSC_XPU float PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light,
                                  float3 wi, const SceneData *data) const;

            NDSC_XPU Spectrum power() const;

            XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s,name:%s, L : %s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this),
                                                   _L.to_string().c_str());
                            })

            CPU_ONLY(static AreaLight create(const LightConfig &config);)
        };
    } //luminous::render
} // luminous::render