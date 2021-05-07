//
// Created by Zero on 2021/4/3.
//

#define GLOBAL extern "C" __global__ void

#include "render/include/shader_include.h"
#include "render/include/interaction.h"
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

extern "C" {
__constant__ luminous::LaunchParams
params;
}

GPU_INLINE luminous::float3 Li(luminous::Ray ray, luminous::Sampler &sampler) {
    using namespace luminous;
    RadiancePRD prd(true, false, false,
                    Spectrum(0.f),
                    Spectrum(1.f),
                    Spectrum(0.f));
    luminous::float3 radiance = make_float3(0.f);
    traceRadiance(params.traversable_handle, ray, &prd);
    return radiance;
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
    prd.done = false;
    prd.count_emitted = true;

    int i = sampler.spp();

    do {



    } while (--i);
    luminous::ray_intersect(params.traversable_handle, ray, &prd);
    film->add_sample(pixel, prd.radiance, 1, params.frame_index);

}

GLOBAL __miss__radiance() {
    luminous::RadiancePRD *prd = getPRD();
    auto data = getSbtData<luminous::MissData>();
    prd->radiance = data.bg_color;
    prd->done = true;
}

GLOBAL __miss__shadow() {

}

GLOBAL __closesthit__radiance() {
    using namespace luminous;
    RadiancePRD *prd = getPRD();
    auto si = getSurfaceInteraction(getClosestHit());
    auto n = si.s_uvn.normal;
    n = (n + 1.f) / 2.f;
    const HitGroupData &data = getSbtData<HitGroupData>();

    prd->radiance = luminous::make_float3(1);
    prd->radiance = n;

    BSDF bsdf = si.material->get_BSDF(si, &data);
    auto color = bsdf.base_color();
    prd->radiance = color;
    prd->hit = true;

//    prd->radiance.print();
}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
