//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "render/include/interaction.h"
#include "render/films/shader_include.h"
#include "render/samplers/shader_include.h"
#include "render/sensors/shader_include.h"
#include "render/light_samplers/shader_include.h"
#include "render/lights/shader_include.h"
#include "render/include/distribution.h"
#include "render/include/shader_include.h"
#include "gpu/framework/optix_params.h"
#include "render/textures/shader_include.h"
#include "render/materials/shader_include.h"
#include "render/bxdfs/shader_include.h"
#include "graphics/lstd/lstd.h"

struct RadiancePRD {
    luminous::Interaction interaction;
    luminous::float3 radiance;
};

static GPU_INLINE void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

static GPU_INLINE void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static GPU_INLINE void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>( occluded ));
}

static GPU_INLINE luminous::uint3 getLaunchIndex() {
    auto idx = optixGetLaunchIndex();
    return luminous::make_uint3(idx.x, idx.y, idx.z);
}

static GPU_INLINE luminous::uint2 getPixelCoords() {
    auto idx = optixGetLaunchIndex();
    return luminous::make_uint2(idx.x, idx.y);
}

template<typename T>
static GPU_INLINE const T &getSbtData() {
    return *reinterpret_cast<T *>(optixGetSbtDataPointer());
}

static GPU_INLINE RadiancePRD *getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD *>(unpackPointer(u0, u1));
}

template<typename... Args>
static GPU_INLINE void trace(OptixTraversableHandle handle,
                             luminous::Ray ray,
                             OptixRayFlags flags,
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
            ray.t_min,
            ray.t_max,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            flags,
            SBToffset,        // SBT offset
            SBTstride,           // SBT stride
            missSBTIndex,        // missSBTIndex
            std::forward<Args>(payload)...);
}

static GPU_INLINE void traceRadiance(OptixTraversableHandle handle,
                                     luminous::Ray ray, RadiancePRD *prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    trace(handle, ray, OPTIX_RAY_FLAG_NONE,
          luminous::RayType::Radiance,        // SBT offset
          luminous::RayType::Count,           // SBT stride
          luminous::RayType::Radiance,        // missSBTIndex
          u0, u1);
}

static GPU_INLINE bool traceOcclusion(OptixTraversableHandle handle, luminous::Ray ray) {
    unsigned int occluded = 0u;
    trace(handle, ray, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
          luminous::RayType::Occlusion,        // SBT offset
          luminous::RayType::Count,           // SBT stride
          luminous::RayType::Occlusion,        // missSBTIndex
          occluded);
    return bool(occluded);
}

static GPU_INLINE luminous::float2 getTriangleBarycentric() {
    float2 barycentric = optixGetTriangleBarycentrics();
    return luminous::make_float2(1 - barycentric.y - barycentric.x, barycentric.x);
}

static GPU_INLINE uint32_t getInstanceId() {
    return optixGetInstanceId();
}

static GPU_INLINE uint32_t getPrimIdx() {
    return optixGetPrimitiveIndex();
}

static GPU_INLINE luminous::ClosestHit getClosestHit() {
    luminous::ClosestHit ret;
    ret.instance_id = getInstanceId();
    ret.triangle_id = getPrimIdx();
    ret.bary = getTriangleBarycentric();
    return ret;
}

static GPU_INLINE luminous::float3 computeShadingNormal(luminous::BufferView<const luminous::float3> normals,
                                                        luminous::TriangleHandle tri,
                                                        luminous::float2 uv) {
    using namespace luminous;
    auto n0 = normals[tri.i];
    auto n1 = normals[tri.j];
    auto n2 = normals[tri.k];
    return triangle_lerp(uv, n0, n1, n2);
}

static GPU_INLINE auto computePosition(luminous::BufferView<const luminous::float3> positions,
                                       luminous::TriangleHandle tri,
                                       luminous::float2 uv) {
    using namespace luminous;
    auto p0 = positions[tri.i];
    auto p1 = positions[tri.j];
    auto p2 = positions[tri.k];
    auto pos = triangle_lerp(uv, p0, p1, p2);
    auto dp02 = p0 - p2;
    auto dp12 = p1 - p2;
    return lstd::make_pair(pos, cross(dp02, dp12));
}

static GPU_INLINE luminous::SurfaceInteraction
computeSurfaceInteraction(luminous::BufferView<const luminous::float3> positions,
                          luminous::BufferView<const luminous::float3> normals,
                          luminous::BufferView<const luminous::float2> tex_coords,
                          luminous::TriangleHandle tri,
                          luminous::float2 uv,
                          luminous::Transform o2w) {
    using namespace luminous;
    SurfaceInteraction si;

    luminous::float2 tex_coord0 = tex_coords[tri.i];
    luminous::float2 tex_coord1 = tex_coords[tri.j];
    luminous::float2 tex_coord2 = tex_coords[tri.k];
    if (tex_coord0.is_zero() && tex_coord1.is_zero() && tex_coord2.is_zero()) {
        tex_coord0 = luminous::make_float2(0, 0);
        tex_coord1 = luminous::make_float2(1, 0);
        tex_coord2 = luminous::make_float2(1, 1);
    }
    si.uv = triangle_lerp(uv, tex_coord0, tex_coord1, tex_coord2);

    luminous::float3 p0 = o2w.apply_point(positions[tri.i]);
    luminous::float3 p1 = o2w.apply_point(positions[tri.j]);
    luminous::float3 p2 = o2w.apply_point(positions[tri.k]);
    luminous::float3 pos = triangle_lerp(uv, p0, p1, p2);
    si.pos = pos;

    // compute geometry uvn
    luminous::float3 dp02 = p0 - p2;
    luminous::float3 dp12 = p1 - p2;
    luminous::float3 ng = cross(dp02, dp12);

    luminous::float2 duv02 = tex_coord0 - tex_coord2;
    luminous::float2 duv12 = tex_coord1 - tex_coord2;
    float det = duv02[0] * duv12[1] - duv02[1] * duv12[0];
    float inv_det = 1 / det;

    luminous::float3 dp_du = (duv12[1] * dp02 - duv02[1] * dp12) * inv_det;
    luminous::float3 dp_dv = (-duv12[0] * dp02 + duv02[0] * dp12) * inv_det;
    si.g_uvn.set(normalize(dp_du), normalize(dp_dv), normalize(ng));

    // compute shading uvn
    luminous::float3 ns = normalize(o2w.apply_normal(computeShadingNormal(normals, tri, uv)));
    luminous::float3 ss = si.g_uvn.dp_du;
    luminous::float3 st = normalize(cross(ns, ss));
    si.s_uvn.set(ss, st, ns);

    return si;
}

static GPU_INLINE luminous::SurfaceInteraction getSurfaceInteraction(luminous::ClosestHit closest_hit) {
    using namespace luminous;
    const HitGroupData &data = getSbtData<HitGroupData>();
    auto si = data.compute_surface_interaction(closest_hit);
    return si;
}
