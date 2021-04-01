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
    luminous::float3 o = luminous::make_float3(0.0, 0.6, -1);
    luminous::float3 d = luminous::make_float3(0, 0, 3);

    luminous::Ray ray;
    luminous::SensorSample ss;
    ss.p_film = pFilm;
    camera->generate_ray(ss, &ray);


    if (ray.direction().has_inf()) {
//        printf("%u, %u\n", idx.x,idx.y);
        return ;
    }

    auto b = traceOcclusion(params.traversable_handle, ray);
    if (b) {
        film->add_sample(pFilm, luminous::make_float3(0,0,1),1);
    } else {
        film->add_sample(pFilm, luminous::make_float3(1,0,0),1);
    }


}

GLOBAL __miss__radiance() {
//    printf("miss radiance\n");
}

GLOBAL __miss__shadow() {
//    printf("miss\n");
}

GLOBAL __closesthit__radiance() {
//    printf("__closesthit__radiance\n");

}

GLOBAL __closesthit__occlusion() {
//    printf("asdf\n");
    setPayloadOcclusion(true);
}
