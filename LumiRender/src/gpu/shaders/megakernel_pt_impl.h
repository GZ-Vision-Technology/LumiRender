//
// Created by Zero on 2021/4/3.
//

#define GLOBAL extern "C" __global__ void

#include "render/include/shader_include.h"
#include "render/samplers/shader_include.h"
#include "render/sensors/shader_include.h"
#include "render/light_samplers/shader_include.h"
#include "render/lights/shader_include.h"
#include "base_libs/sampling/distribution.h"
#include "render/textures/shader_include.h"
#include "render/materials/shader_include.h"
#include "render/scattering/shader_include.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/trace.h"
#include "render/integrators/shader_include.h"
#include "render/scene/shader_include.h"

extern "C" {
__constant__ luminous::LaunchParams
        params;
}

GLOBAL __raygen__rg() {
    // todo implement the simple path tracing to profile the performance
    using namespace luminous;
    luminous::uint2 pixel = getPixelCoords();
    Sensor *camera = params.camera;
    auto film = camera->film();
    Sampler sampler = *params.sampler;
    auto frame_index = params.frame_index;
    sampler.start_pixel_sample(pixel, frame_index, 0);
    auto ss = sampler.sensor_sample(pixel, camera->filter());
    bool debug = false;
    auto[weight, ray] = camera->generate_ray(ss);
    uint spp = sampler.spp();
    PixelInfo pixel_info;
    for (int i = 0; i < spp; ++i) {
        pixel_info += path_tracing(ray, params.traversable_handle, sampler, params.min_depth,
                                   params.max_depth, params.rr_threshold, params.scene_data, debug);
    }
    pixel_info /= float(spp);
    film->add_sample(pixel, pixel_info, weight, frame_index);
}


GLOBAL __closesthit__closest() {
    using namespace luminous;
    HitInfo *hit_info = getPRD<HitInfo>();
    *hit_info = getClosestHit();
}

GLOBAL __closesthit__any() {
    setPayloadOcclusion(true);
}
