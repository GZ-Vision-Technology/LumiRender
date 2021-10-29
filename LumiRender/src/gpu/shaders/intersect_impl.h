//
// Created by Zero on 13/10/2021.
//


#pragma once

#include "gpu/shaders/optix_util.h"
#include "render/integrators/wavefront/params.h"
#include "render/integrators/wavefront/process_queue.cpp"
#include "render/scene/shader_include.h"
#include "render/integrators/wavefront/work_items.h"

#define GLOBAL extern "C" __global__ void

extern "C" {
__constant__ luminous::WavefrontParams
params;
}

GLOBAL __raygen__find_closest() {
    using namespace luminous;
    int task_id = getLaunchIndex();
    if (task_id >= params.ray_queue->size()) {
        return;
    }
    RayWorkItem r = (*params.ray_queue)[task_id];
    Ray ray = r.ray;
    PerRayData prd;
    bool hit = traceClosestHit(params.traversable_handle, ray, &prd);
    if (hit) {
        SurfaceInteraction si = prd.compute_surface_interaction(ray);
        enqueue_item_after_intersect(r, si, params.next_ray_queue,
                                     params.hit_area_light_queue,
                                     params.material_eval_queue);
    } else {
        enqueue_item_after_miss(r, params.escaped_ray_queue);
    }
}

GLOBAL __raygen__occlusion() {
    using namespace luminous;
    int task_id = getLaunchIndex();
    if (task_id >= params.shadow_ray_queue->size()) {
        return;
    }
    ShadowRayWorkItem item = (*params.shadow_ray_queue)[task_id];
    Ray ray = item.ray;
    bool hit = traceAnyHit(params.traversable_handle, ray);
    record_shadow_ray_result(item, params.pixel_sample_state, hit);
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
    const auto &data = getSbtData<SceneData>();
    prd->data = &data;
    prd->hit_point = getClosestHit();
}

GLOBAL __closesthit__any() {
    setPayloadOcclusion(true);
}