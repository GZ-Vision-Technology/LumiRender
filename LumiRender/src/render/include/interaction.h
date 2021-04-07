//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        struct Interaction {
            float3 pos;
            float3 ng;
            float3 ns;
            float2 uv;
            float3 wo;
            float time;
        };

        struct PositionSamplingRecord {
            float3 pos;
            float3 normal;
            float2 uv;
            float PDF_pos;
        };

        struct DirectSamplingRecord : PositionSamplingRecord {
            float3 ref_pos;
            float3 ref_ng;
            // unit vector
            float3 dir;
            float PDF_dir;
            float dist;

            float cos_target_theta() const {
                return dot(-dir, normal);
            }
        };
    }
}