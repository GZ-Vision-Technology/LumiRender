//
// Created by Zero on 2021/4/3.
//

#define GLOBAL extern "C" __global__ void

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
__constant__ luminous::LaunchParams
params;
}


struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};

//struct RadiancePRD
//{
//    // TODO: move some state directly into payload registers?
//    luminous::float4       emitted;
//    luminous::float4       radiance;
//    luminous::float4       attenuation;
//    luminous::float4       origin;
//    luminous::float4       direction;
//
////    float4       emitted;
////    float4       radiance;
////    float4       attenuation;
////    float4       origin;
////    float4       direction;
//
////    vvv4       emitted;
////    vvv4       radiance;
////    vvv4       attenuation;
////    vvv4       origin;
////    vvv4       direction;
//
////    unsigned int seed;
////    int          countEmitted;
////    int          done;
////    int          pad;
////
////    luminous::ClosestHit closest_hit;
////    const void *data{nullptr};
////    luminous::SurfaceInteraction si;
////    lstd::optional<luminous::BSDF> b;
////    luminous::BxDF b;
////    luminous::Frame f;
////
////    NDSC_XPU_INLINE bool is_hit() const {
////        return closest_hit.is_hit();
////    }
//
////    XPU void init_surface_interaction(const HitGroupData *data, Ray ray);
////
////    NDSC_XPU const HitGroupData *hit_group_data() const;
//};

#include "sdk_pt/sutil/vec_math.h"

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
            0,        // SBT offset
            2,           // SBT stride
            0,        // missSBTIndex
            u0, u1 );
}

__device__ void test() {



    luminous::uint2 pixel = getPixelCoords();
    auto camera = params.camera;
    auto film = camera->film();
    auto res = film->resolution();



    const int    w   = res.x;
    const int    h   = res.y;
    auto pos = camera->position();
    const float3 eye = make_float3(pos.x, pos.y, pos.z);
    auto right = camera->right();
    const float3 U   = make_float3(right.x, right.y, right.z);
    auto up = camera->up();

    const float3 V   = make_float3(up.x, up.y, up.z);
    auto forward = camera->forward();
    const float3 W   = make_float3(forward.x, forward.y, forward.z);
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.frame_index;
    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );
    float3 result = make_float3( 0.0f,0.0f,0.0f );



    int i = 1;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        auto subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        auto d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
        ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;

        int depth = 0;
        for( ;; )
        {

            traceRadiance(
                    params.traversable_handle,
                    ::make_float3(278.0f, 273.0f, -900.0f),
            ::make_float3(0,0,1),
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd );
            return;
            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

//            if( prd.done  || depth >= 3 ) // TODO RR, variable for depth
            break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }
    while( --i );
}

GLOBAL __raygen__rg() {
    printf("%p\n", params.frame_buffer);
//    using namespace luminous;
//    luminous::uint2 pixel = getPixelCoords();
//    Sensor *camera = params.camera;
//    Film *film = camera->film();
//    Sampler sampler = *params.sampler;
//    auto frame_index = params.frame_index;
//    sampler.start_pixel_sample(pixel, frame_index, 0);
//    auto ss = sampler.sensor_sample(pixel);
//    bool debug = false;
//    Ray ray(luminous::make_float3(278.0f, 273.0f, -900.0f),
//            luminous::make_float3(0,0,1));
//    float weight = camera->generate_ray(ss, &ray);
//
//    Spectrum L = Li(ray, params.traversable_handle, sampler,
//                    params.max_depth, params.rr_threshold, debug);
//    film->add_sample(pixel, L, weight, frame_index);
}

GLOBAL __miss__radiance() {
    luminous::PerRayData *prd = getPRD();
    const auto &data = getSbtData<luminous::MissData>();
    prd->data = &data;
}

GLOBAL __miss__shadow() {
    setPayloadOcclusion(false);
}

GLOBAL __closesthit__radiance() {
    using namespace luminous;
    PerRayData *prd = getPRD();
    const HitGroupData &data = getSbtData<HitGroupData>();
    prd->data = &data;
    prd->closest_hit = getClosestHit();
}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
