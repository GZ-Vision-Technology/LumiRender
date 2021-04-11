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
static GPU_INLINE T getSbtData() {
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

static GPU_INLINE luminous::Interaction getInteraction(uint32_t instance_id, uint32_t prim_idx,
                                                       luminous::float2 uv) {
    using namespace luminous;
    Interaction interaction;
    HitGroupData data = getSbtData<HitGroupData>();
    uint mesh_idx = data.inst_to_mesh_idx[instance_id];
    luminous::float4x4 mat4x4 = data.transforms[data.inst_to_transform_idx[instance_id]];
    Transform o2w(mat4x4);
    MeshHandle mesh = data.meshes[mesh_idx];
    TriangleHandle tri = data.triangles[mesh.triangle_offset + prim_idx];
    const luminous::float3 *positions = &data.positions[mesh.vertex_offset];
    const luminous::float3 *normals = &data.normals[mesh.vertex_offset];
    const luminous::float2 *tex_coords = &data.tex_coords[mesh.vertex_offset];

    auto n0 = normals[tri.i];
    auto n1 = normals[tri.j];
    auto n2 = normals[tri.k];
    interaction.ns = normalize(o2w.apply_normal(triangle_lerp(uv, n0, n1, n2)));

    auto p0 = positions[tri.i];
    auto p1 = positions[tri.j];
    auto p2 = positions[tri.k];
    auto v01 = p1 - p0;
    auto v02 = p2 - p0;
    interaction.pos = o2w.apply_point(triangle_lerp(uv, p0, p1, p2));
    interaction.ng = normalize(o2w.apply_normal(cross(v01, v02)));

    auto tex_coord0 = tex_coords[tri.i];
    auto tex_coord1 = tex_coords[tri.j];
    auto tex_coord2 = tex_coords[tri.k];

    if (tex_coord0.is_zero() && tex_coord1.is_zero() && tex_coord2.is_zero()) {
        tex_coord0 = luminous::make_float2(0,0);
        tex_coord1 = luminous::make_float2(1,0);
        tex_coord2 = luminous::make_float2(1,1);
    }
    interaction.uv = triangle_lerp(uv, tex_coord0, tex_coord1, tex_coord2);

    return interaction;
}
