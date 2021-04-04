//
// Created by Zero on 2021/4/3.
//

#define GLOBAL extern "C" __global__ void

#include "optix_kernels.h"


extern "C" {
__constant__ luminous::LaunchParams
params;
}

GLOBAL __raygen__rg() {
    using namespace luminous;
    auto pixel = getPixelCoords();
    auto camera = params.camera;
    auto film = camera->film();
    auto sampler = *params.sampler;
    sampler.start_pixel_sample(pixel, 0, 0);
    auto ss = sampler.sensor_sample(pixel);

    Ray ray;
    camera->generate_ray(ss, &ray);
    RadiancePRD prd;
    traceRadiance(params.traversable_handle, ray, &prd);
    film->add_sample(ss.p_film, prd.radiance, 1);
}

GLOBAL __miss__radiance() {
    RadiancePRD *prd = getPRD();
    auto data = getSbtData<luminous::MissData>();
    prd->radiance = data.bg_color;
//    printf("miss radiance\n");
}

GLOBAL __miss__shadow() {
    auto data = getSbtData<luminous::MissData>();
//    data.bg_color.print();
}

GLOBAL __closesthit__radiance() {
//    printf("__closesthit__radiance\n");
    RadiancePRD* prd = getPRD();
    prd->radiance = luminous::make_float3(1);
}

GLOBAL __closesthit__occlusion() {
    auto id = optixGetInstanceId();

    setPayloadOcclusion(true);
}
