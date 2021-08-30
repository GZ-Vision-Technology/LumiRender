//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "soa.h"
#include "base_libs/header.h"
#include "core/backend/device.h"
#include "base_libs/geometry/common.h"
#include "base_libs/math/common.h"
#include "render/bxdfs/bsdf.h"
#include <cuda.h>

namespace luminous {

    inline namespace render {

        LUMINOUS_SOA(float2, x, y)

        LUMINOUS_SOA(float3, x, y, z)

        LUMINOUS_SOA(float4, x, y, z, w)

        LUMINOUS_SOA(Ray, org_x, org_y, org_z, dir_x, dir_y, dir_z, t_min, t_max)

        LUMINOUS_SOA(BSDF, _bxdf, _ng, _shading_frame)

        enum RaySampleFlag {
            hasMedia = 1 << 0,
            hasSubsurface = 1 << 1
        };

        struct RaySamples {
            struct {
                float2 u{};
                float uc{};
            } direct;
            struct {
                float2 u{};
                float uc{}, rr{};
            } indirect;
            RaySampleFlag flag;
        };

        LUMINOUS_SOA(RaySamples, direct, indirect)
    }
}