//
// Created by Zero on 15/10/2021.
//

#include "kernels.cpp"
#include "gpu/shaders/cuda_util.cuh"

#define GLOBAL extern "C" __global__ void

using namespace luminous;

GLOBAL kernel_generate_primary_ray(int task_id, int n_item, int y0, int sample_index,
                                   RayQueue *ray_queue, SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        generate_primary_ray(task_id, n_item, y0, sample_index,
                             ray_queue, pixel_sample_state);
    }
}

GLOBAL kernel_generate_ray_samples(int task_id, int n_item, int sample_index, const RayQueue *ray_queue,
                                   SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        generate_ray_samples(task_id, n_item, sample_index, ray_queue, pixel_sample_state);
    }
}

GLOBAL kernel_process_escape_ray(int task_id, int n_item,
                                 EscapedRayQueue *escaped_ray_queue) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        process_escape_ray(task_id, n_item, escaped_ray_queue);
    }
}

GLOBAL kernel_process_emission(int task_id, int n_item,
                               HitAreaLightQueue *hit_area_light_queue,
                               SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        process_emission(task_id, n_item, hit_area_light_queue, pixel_sample_state);
    }
}

GLOBAL kernel_eval_BSDFs(int task_id, int n_item,
                         MaterialEvalQueue *material_eval_queue) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        eval_BSDFs(task_id, n_item, material_eval_queue);
    }
}