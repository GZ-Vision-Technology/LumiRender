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
#include "render/integrators/shader_include.h"

extern "C" {
__constant__ luminous::LaunchParams
params;
}

namespace luminous {

    XPU_INLINE Spectrum megakernel_pt_Li(luminous::Ray ray, uint64_t scene_handle,
                                         luminous::Sampler &sampler, uint32_t max_depth,
                                         float rr_threshold, bool debug = false) {
        PerRayData prd;
        prd.sampler = &sampler;
        Spectrum L(0.f);
        Spectrum throughput(1.f);
        SurfaceInteraction si;
        int bounces;
        bool found_intersection = luminous::intersect_closest(scene_handle, ray, &prd);
        for (bounces = 0; bounces < max_depth; ++bounces) {
            if (bounces == 0) {
                if (found_intersection) {
                    si = prd.si;
                    L += throughput * si.Le(-ray.direction());
                } else {
                    // nothing to do
                }
            }
            BREAK_IF(!found_intersection)

        }
        return L;
    }
}


GLOBAL __raygen__rg() {
    using namespace luminous;
    luminous::uint2 pixel = getPixelCoords();
    Sensor *camera = params.camera;
    Film *film = camera->film();
    Sampler sampler = *params.sampler;
    auto frame_index = params.frame_index;
    sampler.start_pixel_sample(pixel, frame_index, 0);
    auto ss = sampler.sensor_sample(pixel);
    bool debug = false;
    Ray ray;
    float weight = camera->generate_ray(ss, &ray);
    Spectrum L = Li(ray, params.traversable_handle, sampler,
                    params.max_depth, params.rr_threshold, debug);
    film->add_sample(pixel, L, weight, frame_index);
}

GLOBAL __miss__radiance() {
    luminous::PerRayData *prd = getPRD();
    const auto &data = getSbtData<luminous::MissData>();
    prd->data = &data;
}

GLOBAL __miss__shadow() {
    setPayloadOcclusion(false);
}

GLOBAL __closesthit__radiance() {
    using namespace luminous;
    PerRayData *prd = getPRD();
    const HitGroupData &data = getSbtData<HitGroupData>();
    prd->data = &data;
    prd->closest_hit = getClosestHit();
    Ray ray = getRayInWorld();
    prd->init_surface_interaction(&data, ray);
    Sampler *sampler = prd->sampler;
    SurfaceInteraction &si = prd->si;
    si.init_BSDF(&data);
//    const LightSampler *light_sampler = data.light_sampler;
//    auto op_sampled_light = light_sampler->sample(si, sampler->next_1d());
//    if (op_sampled_light) {
//        auto light = op_sampled_light->light;
//        prd->light_PMF = op_sampled_light->PMF;
//        prd->Ld_sample_light = light.MIS_sample_light()
//    }
}

GLOBAL __closesthit__occlusion() {
    setPayloadOcclusion(true);
}
