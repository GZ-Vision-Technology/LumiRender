//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "light_base.h"

namespace luminous {
    inline namespace render {
        class AreaLight : public LightBase {
        private:
            uint _inst_idx;
            float3 _L;
            bool _two_sided;
            float _inv_area;
        public:
            AreaLight(uint inst_idx, float3 L, bool two_sided = false)
                    : LightBase(LightType::Area),
                      _inst_idx(inst_idx),
                      _L(L),
                      _two_sided(two_sided) {}

            NDSC_XPU float3 L(const Interaction &p_light, float3 w) const;

            GEN_CLASS_NAME(AreaLight)

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU Interaction sample(float u, const HitGroupData * hit_group_data) const;

            NDSC_XPU float PDF_Li(const Interaction &p_ref, const Interaction &p_light) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;

            static AreaLight create(const LightConfig &config);
        };
    } //luminous::render
} // luminous::render