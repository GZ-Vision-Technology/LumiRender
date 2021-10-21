//
// Created by Zero on 14/10/2021.
//

#include "kernels.h"

#ifdef __CUDACC__
#define GLOBAL_PREFIX extern "C" __constant__
#else
#define GLOBAL_PREFIX static
#endif

namespace luminous {
    inline namespace render {


        GLOBAL_PREFIX RTParam *rt_param;

        CPU_ONLY(void set_rt_param(RTParam *param) {
            rt_param = param;
        })

        void generate_primary_ray(int task_id, int n_item, RayQueue *ray_queue, const Sampler *sampler,
                                  SOA<PixelSampleState> *pixel_sample_state) {

        }

        void reset_ray_queue(int task_id, int n_item, RayQueue *ray_queue) {
            ray_queue->reset();
        }

        void reset_queues(int task_id, int n_item, RayQueue *ray_queue,
                          HitAreaLightQueue *hit_area_light_queue,
                          ShadowRayQueue *shadow_ray_queue,
                          EscapedRayQueue *escaped_ray_queue,
                          MaterialEvalQueue *material_eval_queue) {

        }

        void generate_ray_samples(int task_id, int n_item, const RayQueue *ray_queue,
                                  SOA<PixelSampleState> *pixel_sample_state) {

        }

        void process_escape_ray(int task_id, int n_item,
                                EscapedRayQueue *escaped_ray_queue) {

        }

        void process_emission(int task_id, int n_item,
                              HitAreaLightQueue *hit_area_light_queue) {

        }

        void eval_BSDFs(int task_id, int n_item,
                        MaterialEvalQueue *material_eval_queue) {

        }

    }
}