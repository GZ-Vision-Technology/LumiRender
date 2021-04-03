//
// Created by Zero on 2021/4/3.
//

#define GLOBAL extern "C" __global__ void

#include "optix_kernels.h"


extern "C" {
    __constant__ luminous::LaunchParams params;
}

GLOBAL __raygen__rg() {
    auto pixel = getPixelCoords();
    auto pFilm = luminous::make_float2(pixel.x, pixel.y);
    auto camera = params.camera;
    auto sampler = params.sampler;

    auto film = camera->film();

    luminous::Ray ray;
    luminous::SensorSample ss;
    ss.p_film = pFilm;
    camera->generate_ray(ss, &ray);

    auto b = traceOcclusion(params.traversable_handle, ray);
    if (b) {
        film->add_sample(pFilm, luminous::make_float3(0, 0, 1), 1);
    } else {
        film->add_sample(pFilm, luminous::make_float3(1, 0, 0), 1);
    }


}

GLOBAL __miss__radiance() {
//    printf("miss radiance\n");
}

GLOBAL __miss__shadow() {
    auto data = getSbtData < luminous::MissData > ();
//    data.bg_color.print();
}

GLOBAL __closesthit__radiance() {
//    printf("__closesthit__radiance\n");

}

GLOBAL __closesthit__occlusion() {
    auto id = optixGetInstanceId();

    setPayloadOcclusion(true);
}
