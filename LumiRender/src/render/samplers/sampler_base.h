//
// Created by Zero on 2021/3/20.
//


#pragma once

#include "base_libs/geometry/common.h"
#include "core/refl/reflection.h"

namespace luminous {
    inline namespace render {

        struct SensorSample {
            float2 p_film{};
            float2 p_lens{};
            float time{};
            float weight{1.f};

            GEN_STRING_FUNC({
                return string_printf("p_film: %s, p_lens : %s, time: %f, weight : %f",
                                     p_film.to_string().c_str(),
                                     p_lens.to_string().c_str(),
                                     time, weight);
            })
        };

        class SamplerBase : BASE_CLASS() {
        public:
            REFL_CLASS(SamplerBase)
        protected:
            int _spp;
        public:
            LM_XPU explicit SamplerBase(int spp = 1) : _spp(spp) {}

            LM_ND_XPU int spp() const { return _spp; }
        };
    }
}