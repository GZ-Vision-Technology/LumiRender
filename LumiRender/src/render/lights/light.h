//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light_base.h"
#include "point_light.h"
#include "spot_light.h"
#include "area_light.h"
#include "graphics/lstd/lstd.h"
#include "render/include/scene_graph.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Light : public Variant<PointLight, AreaLight> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(Light)

            NDSC_XPU LightType type() const;

            NDSC_XPU const char *name();

            NDSC_XPU bool is_delta() const;

            NDSC_XPU Interaction sample(float u, const HitGroupData *hit_group_data) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU float PDF_Li(const Interaction &ref_p, const Interaction &p_light) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;

            static Light create(const LightConfig &config);
        };
    } // luminous::render
} // luminous