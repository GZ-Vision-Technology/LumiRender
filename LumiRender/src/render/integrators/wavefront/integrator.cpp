//
// Created by Zero on 2021/3/24.
//

#include "integrator.h"

namespace luminous {
    inline namespace gpu {

        WavefrontPT::WavefrontPT(Device *device, Context *context)
                : Integrator(device, context) {

        }

        void WavefrontPT::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            Integrator::init(scene_graph);
            // todo: make this configurable. Base it on the amount of GPU memory?
            int max_samples = 1024 * 1024;
            uint2 res = resolution();
            _scanline_per_pass = std::max(1, int(max_samples / res.x));
            auto n_passes = (res.y + _scanline_per_pass - 1) / _scanline_per_pass;
            _scanline_per_pass = (res.y + n_passes - 1) / n_passes;
            _max_queue_size = res.x * _scanline_per_pass;
            allocate_memory();
        }

        void WavefrontPT::allocate_memory() {
            _ray_queues.emplace_back(_max_queue_size, _device);
            _ray_queues.emplace_back(_max_queue_size, _device);
            _ray_queues.allocate_device(_device);
            _ray_queues.synchronize_to_device();

#define ALLOCATE_AND_SYNCHRONIZE(args)         \
(args).emplace_back(_max_queue_size, _device); \
(args).allocate_device(_device);               \
(args).synchronize_to_device();
            ALLOCATE_AND_SYNCHRONIZE(_shadow_ray_queue)
            ALLOCATE_AND_SYNCHRONIZE(_hit_area_light_queue)
            ALLOCATE_AND_SYNCHRONIZE(_escaped_ray_queue)
            ALLOCATE_AND_SYNCHRONIZE(_material_eval_queue)
            ALLOCATE_AND_SYNCHRONIZE(_pixel_sample_state)
#undef ALLOCATE_AND_SYNCHRONIZE

        }

        void WavefrontPT::render() {

        }
    }
}