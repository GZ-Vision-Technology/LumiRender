//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "uniform.h"
#include "power.h"
#include "bvh.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {

        struct SampledLight {
            LightHandle light;
            float PMF = -1;

            bool valid() const {
                return PMF != -1;
            }

            NDSC std::string to_string() const {
                return string_printf("sampled light :{PMF:%s, light:%s}",
                                     PMF, light.to_string().c_str());
            }
        };


        using lstd::Variant;

        class LightSampler : Variant<UniformLightSampler> {
        private:
            using Variant::Variant;
        public:
            NDSC_XPU SampledLight sample(float u) const;

            NDSC_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU float PMF(const LightHandle &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, float u) const;

            NDSC std::string to_string() const;
        };

    } // luminous::render
} // luminous