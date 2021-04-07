//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "light.h"

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

            NDSC_XPU float3 L(const DirectSamplingRecord &rcd) const;

            NDSC_XPU float3 sample_Li(DirectSamplingRecord *rcd, float2 u) const;

            NDSC_XPU float PDF_Li(const DirectSamplingRecord &rcd) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;
        };
    } //luminous::render
} // luminous::render