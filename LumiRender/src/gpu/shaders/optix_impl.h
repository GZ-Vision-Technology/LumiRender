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
    luminous::uint2 pixel = getPixelCoords();
    Sensor *camera = params.camera;
    Film *film = camera->film();
    Sampler sampler = *params.sampler;
    sampler.start_pixel_sample(pixel, params.frame_index, 0);
    auto ss = sampler.sensor_sample(pixel);

    Ray ray;
    camera->generate_ray(ss, &ray);
    RadiancePRD prd;

    traceRadiance(params.traversable_handle, ray, &prd);
    film->add_sample(pixel, prd.radiance, 1, params.frame_index);

    auto result = luminous::make_float3(0.f);
    for (int i = 0; i < sampler.spp(); ++i) {

    }
}

GLOBAL __miss__radiance() {
    RadiancePRD *prd = getPRD();
    auto data = getSbtData<luminous::MissData>();
    prd->radiance = data.bg_color;
}

GLOBAL __miss__shadow() {

}

GLOBAL __closesthit__radiance() {
    using namespace luminous;
    RadiancePRD *prd = getPRD();
    auto interaction = getInteraction(getInstanceId(), getPrimIdx(), getTriangleBarycentric());
    auto n = interaction.ns;
    n = (n + 1.f) / 2.f;
    HitGroupData data = getSbtData<HitGroupData>();
    auto tex = data.texture_vectors[0];

    TextureEvalContext ctx;
    ctx.uv = interaction.uv;


    prd->radiance = luminous::make_float3(1);
    prd->radiance = n;
    prd->radiance = luminous::make_float3(1-tex.eval(ctx).x);

}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
