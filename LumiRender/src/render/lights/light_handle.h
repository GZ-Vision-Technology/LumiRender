//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light.h"
#include "point_light.h"
#include "spot_light.h"
#include "area_light.h"
#include "mesh_light.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class LightHandle : public Variant<PointLight, AreaLight> {
        private:
            using Variant::Variant;
        public:
            NDSC_XPU LightType type() const;

            NDSC_XPU bool is_delta() const;

            NDSC_XPU float3 sample_Li(DirectSamplingRecord *rcd, float2 u) const;

            NDSC_XPU float PDF_Li(const DirectSamplingRecord &rcd) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;
        };

    } // luminous::render
} // luminous