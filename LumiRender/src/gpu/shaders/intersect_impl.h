//
// Created by Zero on 13/10/2021.
//


#pragma once

#include "gpu/shaders/optix_util.h"
#include "render/integrators/wavefront/params.h"
#include "render/integrators/wavefront/process_queue.cpp"
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
        return ;
    }
    RayWorkItem r = (*params.ray_queue)[task_id];
    Ray ray = r.ray;
    PerRayData prd;
    bool hit = traceClosestHit(params.traversable_handle, ray, &prd);
    printf("%d \n", int(hit));
}

GLOBAL __raygen__occlusion() {
    if (getLaunchIndex() >= params.ray_queue->size()) {
        return ;
    }
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