//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "scene_graph.h"

namespace luminous {
    inline namespace render {

        class SamplerBase : public IObject {
        protected:
            int _spp;
        public:
            explicit SamplerBase(int spp = 1) : _spp(spp) {}

            NDSC int spp() const { return _spp; }
        };

        class LCGSampler;

        class PCGSampler;

        using lstd::Variant;

        class SamplerHandle : public Variant<LCGSampler *, PCGSampler *> {
            using Variant::Variant;
        public:
            NDSC XPU int spp() const;

            XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            XPU const char *name();

            NDSC XPU float next_1d();

            NDSC XPU float2 next_2d();

            NDSC std::string to_string();

            static SamplerHandle create(const Config &config);
        };
    }
}