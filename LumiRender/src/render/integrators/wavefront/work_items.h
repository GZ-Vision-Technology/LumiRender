//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "soa.h"
#include "base_libs/header.h"
#include "core/backend/device.h"
#include "base_libs/geometry/common.h"
#include "base_libs/math/common.h"

namespace luminous {


    inline namespace render {

        LUMINOUS_SOA(float2, x, y)
        LUMINOUS_SOA(float3, x, y, z)
        LUMINOUS_SOA(float4, x, y, z, w)

    }
}