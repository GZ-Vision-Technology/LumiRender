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

//    float3 o = make_float3(0,0,0);
//    float3 d = make_float3(0,0,1);
//
//    RadiancePRD prd;
//
//    traceRadiance(
//            params.traversable_handle,
//            o,
//            d,
//            0.01f,  // tmin       // TODO: smarter offset
//            1e16f,  // tmax
//            &prd );
}

GLOBAL __miss__radiance() {

}

GLOBAL __miss__shadow() {

}

GLOBAL __closesthit__radiance() {
    printf("asdf\n");
}

GLOBAL __closesthit__occlusion() {

}
