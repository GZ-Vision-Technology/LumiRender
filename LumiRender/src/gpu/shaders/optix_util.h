//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "gpu/accel/optix_params.h"


static LM_GPU_INLINE void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

static LM_GPU_INLINE void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static LM_GPU_INLINE void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>( occluded ));
}

static LM_GPU_INLINE luminous::uint3 getLaunchIndex3D() {
    auto idx = optixGetLaunchIndex();
    return luminous::make_uint3(idx.x, idx.y, idx.z);
}

static LM_GPU_INLINE luminous::uint2 getPixelCoords() {
    auto idx = optixGetLaunchIndex();
    return luminous::make_uint2(idx.x, idx.y);
}

static LM_GPU_INLINE int getLaunchIndex() {
    auto idx = optixGetLaunchIndex();
    return idx.x;
}

template<typename T>
static LM_GPU_INLINE const T &getSbtData() {
    return *reinterpret_cast<T *>(optixGetSbtDataPointer());
}

template<typename T = luminous::PerRayData>
static LM_GPU_INLINE T *getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

template<typename... Args>
static LM_GPU_INLINE void trace(OptixTraversableHandle handle,
                                luminous::Ray ray,
                                uint32_t flags,
                                uint32_t SBToffset,
                                uint32_t SBTstride,
                                uint32_t missSBTIndex,
                                Args &&... payload) {
    float3 origin = make_float3(ray.org_x, ray.org_y, ray.org_z);
    float3 direction = make_float3(ray.dir_x, ray.dir_y, ray.dir_z);

    optixTrace(
            handle,
            origin,
            direction,
            0,
            ray.t_max,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            flags,
            SBToffset,        // SBT offset
            SBTstride,           // SBT stride
            missSBTIndex,        // missSBTIndex
            std::forward<Args>(payload)...);
}

static LM_GPU_INLINE bool traceClosestHit(OptixTraversableHandle handle,
                                          luminous::Ray ray, luminous::PerRayData *prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          luminous::RayType::ClosestHit,        // SBT offset
          luminous::RayType::Count,           // SBT stride
          luminous::RayType::ClosestHit,        // missSBTIndex
          u0, u1);
    return prd->is_hit();
}

static LM_GPU_INLINE bool traceAnyHit(OptixTraversableHandle handle, luminous::Ray ray) {
    unsigned int occluded = 1u;
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT
                       | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                       | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
          luminous::RayType::AnyHit,        // SBT offset
          luminous::RayType::Count,           // SBT stride
          luminous::RayType::AnyHit,        // missSBTIndex
          occluded);
    return bool(occluded);
}

static LM_GPU_INLINE luminous::float2 getTriangleBarycentric() {
    float2 barycentric = optixGetTriangleBarycentrics();
    return luminous::make_float2(1 - barycentric.y - barycentric.x, barycentric.x);
}

static LM_GPU_INLINE luminous::Ray getRayInWorld() {
    float3 d = optixGetWorldRayDirection();
    auto dir = luminous::make_float3(d.x, d.y, d.z);
    float3 o = optixGetWorldRayOrigin();
    auto org = luminous::make_float3(o.x, o.y, o.z);
    float t_min = optixGetRayTmin();
    float t_max = optixGetRayTmax();
    return luminous::Ray(org, dir, t_max);
}

static LM_GPU_INLINE uint32_t getInstanceId() {
    return optixGetInstanceId();
}

static LM_GPU_INLINE uint32_t getPrimIdx() {
    return optixGetPrimitiveIndex();
}

static LM_GPU_INLINE luminous::HitPoint getClosestHit() {
    luminous::HitPoint ret;
    ret.instance_id = getInstanceId();
    ret.triangle_id = getPrimIdx();
    ret.bary = getTriangleBarycentric();
    return ret;
}
