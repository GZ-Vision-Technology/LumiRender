//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "light_base.h"
#include "parser/config.h"

namespace luminous {
    inline namespace render {
        class AreaLight : BASE_CLASS(LightBase) {
        public:
            REFL_CLASS(AreaLight)

        private:
            uint _inst_idx{};
            float3 L{};
            bool _two_sided{};
        public:
            CPU_ONLY(explicit AreaLight(const LightConfig &config)
                    : AreaLight(config.instance_idx, config.emission,
                                config.two_sided) {})

            AreaLight(uint inst_idx, float3 L, bool two_sided);

            LM_ND_XPU Spectrum radiance(const LightEvalContext &lec, float3 w,
                                        const SceneData *scene_data) const;

            LM_ND_XPU Spectrum radiance(float2 uv, float3 ng, float3 w, const SceneData *scene_data) const;

            LM_ND_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            LM_ND_XPU LightEvalContext sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            LM_ND_XPU float PDF_Li(const LightSampleContext &p_ref, const LightEvalContext &p_light,
                                   float3 wi_un, const SceneData *data) const;

            LM_ND_XPU Spectrum power() const;

            LM_XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s,name:%s, L : %s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this),
                                                   L.to_string().c_str());
                            })
        };
    } //luminous::render
} // luminous::render