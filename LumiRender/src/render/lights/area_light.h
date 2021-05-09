//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "light_base.h"
#include "render/include/trace.h"

namespace luminous {
    inline namespace render {
        class AreaLight : public LightBase {
        private:
            uint _inst_idx;
            float3 _L;
            bool _two_sided;
            float _area;
        public:
            AreaLight(uint inst_idx, float3 L, float area, bool two_sided = false)
                    : LightBase(LightType::Area),
                      _inst_idx(inst_idx),
                      _L(L),
                      _area(area),
                      _two_sided(two_sided) {}

            NDSC_XPU Spectrum L(const SurfaceInteraction &p_light, float3 w) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU SurfaceInteraction sample(float2 u, const HitGroupData *hit_group_data) const;

            NDSC_XPU float PDF_dir(const Interaction &p_ref, const SurfaceInteraction &p_light) const;

            NDSC_XPU Spectrum power() const;

            NDSC std::string to_string() const;

            static AreaLight create(const LightConfig &config);
        };
    } //luminous::render
} // luminous::render