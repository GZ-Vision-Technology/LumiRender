//
// Created by Zero on 2021/2/14.
//

#define GLOBAL extern "C" __global__ void

#include "optix_kernels.h"


extern "C" {
__constant__ luminous::LaunchParams params;
}

GLOBAL __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    auto pFilm = luminous::make_float2(idx.x, idx.y);
    auto camera = params.camera;
    auto film = camera->film();
////    printf("%s\n", film->name());
    film->add_sample(pFilm, luminous::make_float3(1.f,0.2f,1.f), 1.f);

    float3 o = make_float3(0,0.25,-1);
    float3 d = make_float3(0,0,1);

    RadiancePRD prd;
//    printf("%llu \n", params.traversable_handle);
    traceRadiance(
            params.traversable_handle,
            o,
            d,
            -0.01f,  // tmin       // TODO: smarter offset
            100,  // tmax
            &prd );
}

GLOBAL __miss__radiance() {
    printf("miss radiance\n");
}

GLOBAL __miss__shadow() {
    printf("miss\n");
}

GLOBAL __closesthit__radiance() {
    printf("asdf\n");
}

GLOBAL __closesthit__occlusion() {
    printf("asdf\n");
}
