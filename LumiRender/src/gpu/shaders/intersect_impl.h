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
    HitContext hit_ctx;
    bool hit = traceClosestHit(params.traversable_handle, ray, &hit_ctx.hit_info);
    if (hit) {
        enqueue_item_after_intersect(r, hit_ctx, params.next_ray_queue,
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
    luminous::HitContext *hit_ctx = getPRD();
    const auto &data = getSbtData<luminous::SceneData>();
    hit_ctx->data = &data;
}

GLOBAL __miss__any() {
    setPayloadOcclusion(false);
}

GLOBAL __closesthit__closest() {
    using namespace luminous;
    HitContext *hit_ctx = getPRD();
    const auto &data = getSbtData<SceneData>();
    hit_ctx->data = &data;
    hit_ctx->hit_info = getClosestHit();
}

GLOBAL __closesthit__any() {
    setPayloadOcclusion(true);
}