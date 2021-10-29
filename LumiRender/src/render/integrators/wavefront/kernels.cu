//
// Created by Zero on 15/10/2021.
//

#include "kernels.cpp"
#include "gpu/shaders/cuda_util.cuh"

#define GLOBAL extern "C" __global__ void

using luminous::PixelSampleState;

#define DEFINE_KERNEL(funcName, Args, CallArgs)    \
GLOBAL kernel_##funcName Args {                    \
        task_id = task_id_g3_b3();                 \
        if (task_id < n_item) {                    \
            luminous::##funcName CallArgs;         \
        }                                          \
    }

GLOBAL kernel_generate_primary_ray(int task_id, int n_item, int y0, int sample_index,
                                   luminous::RayQueue *ray_queue,
                                   luminous::SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        luminous::generate_primary_ray(task_id, n_item, y0, sample_index,
                                       ray_queue, pixel_sample_state);
    }
}

GLOBAL kernel_generate_ray_samples(int task_id, int n_item, int sample_index,
                                   const luminous::RayQueue *ray_queue,
                                   luminous::SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        luminous::generate_ray_samples(task_id, n_item, sample_index, ray_queue, pixel_sample_state);
    }
}

GLOBAL kernel_process_escape_ray(int task_id, int n_item,
                                 luminous::EscapedRayQueue *escaped_ray_queue,
                                 luminous::SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        luminous::process_escape_ray(task_id, n_item, escaped_ray_queue, pixel_sample_state);
    }
}

GLOBAL kernel_process_emission(int task_id, int n_item,
                               luminous::HitAreaLightQueue *hit_area_light_queue,
                               luminous::SOA<PixelSampleState> *pixel_sample_state) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        luminous::process_emission(task_id, n_item, hit_area_light_queue, pixel_sample_state);
    }
}


GLOBAL kernel_estimate_direct_lighting(int task_id, int n_item,
                                       luminous::ShadowRayQueue *shadow_ray_queue,
                                       luminous::RayQueue *next_ray_queue,
                                       luminous::MaterialEvalQueue *material_eval_queue) {
    task_id = task_id_g3_b3();
    if (task_id < n_item) {
        luminous::estimate_direct_lighting(task_id, n_item, shadow_ray_queue, next_ray_queue, material_eval_queue);
    }
}