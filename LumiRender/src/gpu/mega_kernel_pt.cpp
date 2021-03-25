//
// Created by Zero on 2021/3/23.
//

#include "mega_kernel_pt.h"

namespace luminous {
    inline namespace gpu {

        void MegaKernelPT::init_launch_params() {
            LaunchParams lp{};
            lp.camera = _camera.device_data();
            _launch_params.reset(lp, _device);
            _launch_params.synchronize_to_gpu();
        }

        void MegaKernelPT::init(const std::shared_ptr<SceneGraph> &scene_graph,
                                SensorHandle *camera) {
            _scene = make_unique<GPUScene>(_device);
            _scene->init(scene_graph);
            _camera.reset(camera, _device);
            auto sampler = SamplerHandle::create(scene_graph->sampler_config);
            _sampler.reset(sampler, _device);

            init_launch_params();
        }

        void MegaKernelPT::render() {
            auto res = _camera->resolution();
            _scene->launch(res, _launch_params);
        }

        void MegaKernelPT::synchronize_to_gpu() {
            _camera.synchronize_to_gpu();
        }

        void MegaKernelPT::update() {
            synchronize_to_gpu();
        }
    }
}