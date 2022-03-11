//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/math/rng.h"
#include "parser/config.h"
#include "sampler_base.h"
#include "core/concepts.h"

#ifndef __CUDACC__
#include <random>
#endif

namespace luminous {
    inline namespace render {

        class DebugSampler: public SamplerBase {

            DECLARE_REFLECTION(DebugSampler, SamplerBase)

        private:
            LCG<> _rng;
        public:
            CPU_ONLY(explicit DebugSampler(const SamplerConfig &config) : DebugSampler(config.spp) {})

            LM_XPU explicit DebugSampler(int spp = 1) : SamplerBase(spp) {}

            LM_XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            LM_ND_XPU int compute_dimension(int depth) const;

            LM_NODISCARD LM_XPU float next_1d();

            LM_NODISCARD LM_XPU float2 next_2d();

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s:{spp=%d}", type_name(this), spp())
                            })
        };

        class PCGSampler : public SamplerBase {

            DECLARE_REFLECTION(PCGSampler, SamplerBase)

        private:
            RNG _rng;
            int _seed{};
        public:
            CPU_ONLY(explicit PCGSampler(const SamplerConfig &sc) : PCGSampler(sc.spp) {})

            LM_XPU explicit PCGSampler(int spp = 1) : SamplerBase(spp) {}

            LM_XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension);

            LM_ND_XPU int compute_dimension(int depth) const;

            LM_NODISCARD LM_XPU float next_1d();

            LM_NODISCARD LM_XPU float2 next_2d();

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("%s:{spp=%d}", type_name(this), spp())
                            })
        };
    }
}