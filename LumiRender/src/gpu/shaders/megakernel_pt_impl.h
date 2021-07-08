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
    int flag = 0;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};

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

GLOBAL __raygen__rg() {
    using namespace luminous;
    luminous::uint2 pixel = getPixelCoords();
    Sensor *camera = params.camera;
    Film *film = camera->film();
    Sampler sampler = *params.sampler;
    auto frame_index = params.frame_index;
    sampler.start_pixel_sample(pixel, frame_index, 0);
    auto ss = sampler.sensor_sample(pixel);
    bool debug = false;
    Ray ray(luminous::make_float3(278.0f, 273.0f, -900.0f),
            luminous::make_float3(0,0,1));
    float weight = camera->generate_ray(ss, &ray);
    uint spp = sampler.spp();
    Spectrum L(0.f);
    for (int i = 0; i < spp; ++i) {
        L += Li(ray, params.traversable_handle, sampler,
                params.max_depth, params.rr_threshold, debug);
    }
    L = L / float(spp);
    film->add_sample(pixel, L, weight, frame_index);
}

GLOBAL __miss__radiance() {
    luminous::PerRayData *prd = getPRD();
    const auto &data = getSbtData<luminous::MissData>();
    prd->data = &data;
}

GLOBAL __miss__shadow() {
    setPayloadOcclusion(false);
}

static __forceinline__ __device__ RadiancePRD* getPRD2()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
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
