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
#include "base_libs/sampling/distribution.h"
#include "render/textures/shader_include.h"
#include "render/materials/shader_include.h"
#include "render/bxdfs/shader_include.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/trace.h"
#include "render/integrators/shader_include.h"
#include "render/scene/shader_include.h"

extern "C" {
__constant__ luminous::LaunchParams
params;
}

GLOBAL __raygen__rg() {
    using namespace luminous;
    luminous::uint2 pixel = getPixelCoords();
    Sensor *camera = params.camera;
    auto film = camera->film();
    Sampler sampler = *params.sampler;
    auto frame_index = params.frame_index;
    // todo single frame multi sample and single sample multi frame can not reach a same result
    sampler.start_pixel_sample(pixel, frame_index, 0);
    auto ss = sampler.sensor_sample(pixel);
    bool debug = pixel.x == 383 && pixel.y == 383;
    Ray ray{};
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

GLOBAL __miss__closest() {
    luminous::PerRayData *prd = getPRD();
    const auto &data = getSbtData<luminous::SceneData>();
    prd->data = &data;
}

GLOBAL __miss__any() {
    setPayloadOcclusion(false);
}

GLOBAL __closesthit__closest() {
    using namespace luminous;
    PerRayData *prd = getPRD();
    const SceneData &data = getSbtData<SceneData>();
    prd->data = &data;
    prd->hit_point = getClosestHit();
}

GLOBAL __closesthit__any() {
    setPayloadOcclusion(true);
}
