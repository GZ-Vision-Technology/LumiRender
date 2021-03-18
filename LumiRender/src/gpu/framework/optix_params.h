//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace gpu {
        struct Params {
            OptixTraversableHandle traversable_handle;
            uint frame_index;
            float4 *accum_buffer;
            uchar4 *frame_buffer;
            uint width;
            uint height;
            uint samples_per_launch;
        };

        struct RayGenData {

        };


        struct MissData {
            float4 bg_color;
        };

        struct HitGroupData {

        };

        template<typename T>
        struct Record {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            T data;
        };

        enum RayType {
            Radiance = 0,
            Occlusion = 1,
            Count
        };

        using RayGenRecord = Record<RayGenData>;
        using MissRecord = Record<MissData>;
        using HitGroupRecord = Record<HitGroupData>;
    }
}