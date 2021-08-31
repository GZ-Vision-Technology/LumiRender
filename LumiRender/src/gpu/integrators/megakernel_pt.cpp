//
// Created by Zero on 2021/3/23.
//

#include "megakernel_pt.h"
#include "gpu/gpu_scene.h"

namespace luminous {
    inline namespace gpu {

        MegakernelPT::MegakernelPT(const SP<Device> &device, Context *context)
                : GPUIntegrator(device, context) {}

        void MegakernelPT::init_launch_params() {
            LaunchParams lp{};

            lp.sampler = _sampler.device_data();
            lp.camera = _camera.device_data();
            lp.frame_index = 0u;
            lp.max_depth = _max_depth;
            lp.rr_threshold = _rr_threshold;
            _launch_params.reset(&lp, _device);
        }

        void MegakernelPT::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            _scene = std::make_shared<GPUScene>(_device, _context);
            init_with_config(scene_graph->integrator_config);
            _scene->init(scene_graph);
            auto camera = Sensor::create(scene_graph->sensor_config);
            _camera.reset(&camera, _device);
            auto sampler = Sampler::create(scene_graph->sampler_config);
            _sampler.reset(&sampler, _device);
            init_launch_params();
        }

        void MegakernelPT::render() {
            auto res = _camera->resolution();
            scene<GPUScene>()->launch(res, _launch_params);
            _launch_params->frame_index += 1;
        }

        void MegakernelPT::synchronize_to_gpu() {
            _camera.synchronize_to_gpu();
            _sampler.synchronize_to_gpu();
        }

        void MegakernelPT::update() {
            _launch_params->frame_index = 0u;
            synchronize_to_gpu();
        }

    }
}