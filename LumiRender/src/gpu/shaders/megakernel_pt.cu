
#if 0
#include "megakernel_pt_impl.h"

#else
//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <optix.h>

#include "optixPathTracer.h"

#include <sutil/vec_math.h>


#include "render/include/shader_include.h"
#include "render/films/shader_include.h"
#include "render/samplers/shader_include.h"
#include "render/sensors/shader_include.h"
#include "render/light_samplers/shader_include.h"
#include "render/lights/shader_include.h"
#include "render/include/distribution.h"
#include "render/textures/shader_include.h"
#include "render/materials/shader_include.h"
#include "render/bxdfs/shader_include.h"
#include "graphics/lstd/lstd.h"
#include "render/include/trace.h"
#include "render/integrators/shader_include.h"


extern "C" {
__constant__ Params params;
}



//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct  vvv4 {
    float a;
    float a0;
    float a1;
    float a2;
};

//struct alignas(sizeof(float) * 4) vvv4 {
//    float a;
//    float a0;
//    float a1;
//    float a2;
//};

//#include "render/bxdfs/shader_include.h"

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    luminous::float4       emitted;
    luminous::float4       radiance;
    luminous::float4       attenuation;
    luminous::float4       origin;
    luminous::float4       direction;

//    float4       emitted;
//    float4       radiance;
//    float4       attenuation;
//    float4       origin;
//    float4       direction;

//    vvv4       emitted;
//    vvv4       radiance;
//    vvv4       attenuation;
//    vvv4       origin;
//    vvv4       direction;

//    unsigned int seed;
//    int          countEmitted;
//    int          done;
//    int          pad;
//
//    luminous::ClosestHit closest_hit;
//    const void *data{nullptr};
//    luminous::SurfaceInteraction si;
//    lstd::optional<luminous::BSDF> b;
    luminous::BxDF b;
    luminous::Frame f;
//
//    NDSC_XPU_INLINE bool is_hit() const {
//        return closest_hit.is_hit();
//    }

//    XPU void init_surface_interaction(const HitGroupData *data, Ray ray);
//
//    NDSC_XPU const HitGroupData *hit_group_data() const;
};


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

//static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
//{
//    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
//    void*           ptr = reinterpret_cast<void*>( uptr );
//    return ptr;
//}
//
//
//static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
//{
//    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
//    i0 = uptr >> 32;
//    i1 = uptr & 0x00000000ffffffff;
//}


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
)
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
)
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded );
    return occluded;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
//    RadiancePRD prd;
//    prd.emitted      = make_float3(0.f);
//    prd.radiance     = make_float3(0.f);
//    prd.attenuation  = make_float3(1.f);
//    prd.countEmitted = true;
//    prd.done         = false;
//    traceRadiance(
//            params.traversable_handle,
//            make_float3(278.0f, 273.0f, -900.0f),
//            make_float3(0,0,1),
//            0.01f,  // tmin       // TODO: smarter offset
//            1e16f,  // tmax
//            &prd );
//    return ;
//    using namespace luminous;
//    luminous::uint2 pixel = getPixelCoords();
//    Sensor *camera = params.camera;
//    Film *film = camera->film();
//    Sampler sampler = *params.sampler;
//    auto frame_index = params.frame_index;
//    sampler.start_pixel_sample(pixel, frame_index, 0);
//    auto ss = sampler.sensor_sample(pixel);
//    bool debug = false;
//    Ray ray;
//    float weight = camera->generate_ray(ss, &ray);

    RadiancePRD prd{};
    traceRadiance(
            params.traversable_handle,
            ::make_float3(278.0f, 273.0f, -900.0f),
            ::make_float3(0,0,1),
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &prd );
//    PerRayData prd;
//    ray = luminous::Ray(luminous::make_float3(278.0f, 273.0f, -900.0f),
//                        luminous::make_float3(0,0,1),1e16f,0.01f);
//    luminous::intersect_closest(params.traversable_handle, ray, &prd);
}


extern "C" __global__ void __miss__radiance()
{

}

extern "C" __global__ void __miss__shadow() {

}

extern "C" __global__ void __closesthit__occlusion()
{

}


extern "C" __global__ void __closesthit__radiance()
{
//    int prim_idx = optixGetPrimitiveIndex();
//    printf("%d \n", prim_idx);
}

#endif
