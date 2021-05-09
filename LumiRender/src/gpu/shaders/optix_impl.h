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
    RadiancePRD prd;
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

    int i = sampler.spp();

    luminous::intersect_closest(params.traversable_handle, ray, &prd);

    film->add_sample(pixel, prd.is_hit() ? 1 : 0, 1, params.frame_index);

}

GLOBAL __miss__radiance() {
    luminous::RadiancePRD *prd = getPRD();
    const auto &data = getSbtData<luminous::MissData>();
    prd->miss_data = &data;
}

GLOBAL __miss__shadow() {

}

GLOBAL __closesthit__radiance() {
    using namespace luminous;
    RadiancePRD *prd = getPRD();
    const HitGroupData &data = getSbtData<HitGroupData>();
    prd->hit_group_data = &data;
    prd->closest_hit = getClosestHit();
}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
