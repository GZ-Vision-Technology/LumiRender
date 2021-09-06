//
// Created by Zero on 2021/3/23.
//

#include "megakernel_pt.h"
#include "gpu/gpu_scene.h"

namespace luminous {
    inline namespace gpu {

        MegakernelPT::MegakernelPT(Device *device, Context *context)
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
            Integrator::init(scene_graph);
            _scene->init_accel<MegakernelOptixAccel>();
            init_on_device();
            init_launch_params();
        }

        void MegakernelPT::render() {
            auto res = _camera->resolution();
            _scene->accel<MegakernelOptixAccel>()->launch(res, _launch_params);
            _launch_params->frame_index += 1;
        }

        void MegakernelPT::synchronize_to_gpu() {
            _camera.synchronize_to_device();
            _sampler.synchronize_to_device();
        }

        void MegakernelPT::update() {
            _launch_params->frame_index = 0u;
            synchronize_to_gpu();
        }

    }
}