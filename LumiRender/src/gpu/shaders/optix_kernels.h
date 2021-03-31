//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "render/include/interaction.h"
#include "render/films/shader_include.h"
#include "render/samplers/shader_include.h"
#include "render/sensors/shader_include.h"
#include "gpu/framework/optix_params.h"

struct RadiancePRD {
    // TODO: move some state directly into payload registers?
    int pad;
};

static __forceinline__ __device__

void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

static __forceinline__ __device__

void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__

void traceRadiance(
        OptixTraversableHandle handle,
        float3 ray_origin,
        float3 ray_direction,
        float tmin,
        float tmax,
        RadiancePRD *prd
) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            luminous::RayType::Radiance,        // SBT offset
            luminous::RayType::Count,           // SBT stride
            luminous::RayType::Radiance,        // missSBTIndex
            u0, u1);
}

static __forceinline__ GPU void traceRadiance(OptixTraversableHandle handle,
                                              luminous::Ray ray, RadiancePRD *prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    float3 origin = make_float3(ray.org_x, ray.org_y, ray.org_z);
    float3 direction = make_float3(ray.dir_x, ray.dir_y, ray.dir_z);
    optixTrace(
            handle,
            origin,
            direction,
            ray.t_min,
            ray.t_max,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            luminous::RayType::Radiance,        // SBT offset
            luminous::RayType::Count,           // SBT stride
            luminous::RayType::Radiance,        // missSBTIndex
            u0, u1);
}

static __forceinline__ GPU bool traceOcclusion(OptixTraversableHandle handle, luminous::Ray ray) {
    unsigned int occluded = 0u;
    float3 origin = make_float3(ray.org_x, ray.org_y, ray.org_z);
    float3 direction = make_float3(ray.dir_x, ray.dir_y, ray.dir_z);
    optixTrace(
            handle,
            origin,
            direction,
            ray.t_min,
            ray.t_max,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            luminous::RayType::Occlusion,        // SBT offset
            luminous::RayType::Count,           // SBT stride
            luminous::RayType::Occlusion,        // missSBTIndex
            occluded);
    return occluded;
}

static __forceinline__ __device__

bool traceOcclusion(
        OptixTraversableHandle handle,
        float3 ray_origin,
        float3 ray_direction,
        float tmin,
        float tmax
) {
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            luminous::RayType::Occlusion,        // SBT offset
            luminous::RayType::Count,           // SBT stride
            luminous::RayType::Occlusion,        // missSBTIndex
            occluded);
    return occluded;
}