//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "../include/scene_graph.h"
#include "independent.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class SamplerHandle : public Variant<LCGSampler, PCGSampler> {
        private:
            using Variant::Variant;
        public:
            NDSC XPU int spp() const;

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            XPU SensorSample sensor_sample(int2 p_raster);

            NDSC_XPU const char *name();

            NDSC_XPU float next_1d();

            NDSC_XPU float2 next_2d();

            NDSC std::string to_string();

            static SamplerHandle create(const SamplerConfig &config);
        };
    }
}