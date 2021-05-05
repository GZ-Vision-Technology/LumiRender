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
            float _inv_area;
        public:
            AreaLight(uint inst_idx, float3 L, float area, bool two_sided = false)
                    : LightBase(LightType::Area),
                      _inst_idx(inst_idx),
                      _L(L),
                      _inv_area(1.f / area),
                      _two_sided(two_sided) {}

            NDSC_XPU float3 L(const SurfaceInteraction &p_light, float3 w) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU SurfaceInteraction sample(float2 u, const HitGroupData *hit_group_data) const;

            NDSC_XPU LightLiSample sample_Li(float2 u, LightLiSample lls, const HitGroupData *hit_group_data) const;

            NDSC_XPU float PDF_pos() const {
                return _inv_area;
            }

            NDSC_XPU float PDF_dir(const Interaction &p_ref, const SurfaceInteraction &p_light) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;

            static AreaLight create(const LightConfig &config);
        };
    } //luminous::render
} // luminous::render