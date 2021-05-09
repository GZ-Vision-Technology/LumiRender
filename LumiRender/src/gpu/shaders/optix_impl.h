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

extern "C" {
__constant__ luminous::LaunchParams
params;
}

GPU_INLINE luminous::Spectrum Li(luminous::Ray ray, luminous::Sampler &sampler) {
    using namespace luminous;
    RadiancePRD prd(true, false, false,
                    Spectrum(0.f),
                    Spectrum(1.f),
                    Spectrum(0.f));
    Spectrum radiance(0.f);
//    for (int bounce = 0; ; ++bounce) {
//
//    }

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
    luminous::intersect_closest(params.traversable_handle, ray, &prd);

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
    SurfaceInteraction si = getSurfaceInteraction(getClosestHit());
    const HitGroupData &data = getSbtData<HitGroupData>();
    prd->interaction = si;
    prd->data = &data;
    BSDF bsdf = si.material->get_BSDF(si, &data);
    auto color = bsdf.base_color();
    prd->radiance = color;
    prd->hit = true;
}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
