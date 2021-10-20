//
// Created by Zero on 15/10/2021.
//

#include "kernels.cpp"
#include "gpu/shaders/cuda_util.cuh"

#define GLOBAL extern "C" __global__ void

using namespace luminous;

GLOBAL kernel_generate_primary_ray(int task_id, int n_item, RayQueue *ray_queue, const Sampler *sampler,
                                   SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    generate_primary_ray(task_id, n_item, ray_queue, sampler, pixel_sample_state);
}

GLOBAL kernel_reset_ray_queue(int task_id, int n_item, RayQueue *ray_queue) {
    task_id = task_id_g3_b3();
    reset_ray_queue(task_id, n_item, ray_queue);
}

GLOBAL kernel_reset_queues(int task_id, int n_item, RayQueue *ray_queue,
                         HitAreaLightQueue *hit_area_light_queue,
                         ShadowRayQueue *shadow_ray_queue,
                         EscapedRayQueue *escaped_ray_queue,
                         MaterialEvalQueue *material_eval_queue) {
    task_id = task_id_g3_b3();
    reset_queues(task_id, n_item, ray_queue, hit_area_light_queue, shadow_ray_queue,
                 escaped_ray_queue, material_eval_queue);
}

GLOBAL kernel_generate_ray_samples(int task_id, int n_item, const RayQueue *ray_queue,
                                 SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    generate_ray_samples(task_id, n_item, ray_queue, pixel_sample_state);
}

GLOBAL kernel_process_escape_ray(int task_id, int n_item,
                               EscapedRayQueue *escaped_ray_queue) {
    task_id = task_id_g3_b3();
    process_escape_ray(task_id, n_item, escaped_ray_queue);
}

GLOBAL kernel_process_emission(int task_id, int n_item,
                             HitAreaLightQueue *hit_area_light_queue) {
    task_id = task_id_g3_b3();
    process_emission(task_id, n_item, hit_area_light_queue);
}

GLOBAL kernel_eval_BSDFs(int task_id, int n_item,
                       MaterialEvalQueue *material_eval_queue) {
    task_id = task_id_g3_b3();
    eval_BSDFs(task_id, n_item, material_eval_queue);
}