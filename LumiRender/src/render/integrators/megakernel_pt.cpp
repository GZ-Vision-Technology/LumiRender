//
// Created by Zero on 2021/3/23.
//

#include "megakernel_pt.h"
#include "render/scene/gpu_scene.h"
#include "util/progressreporter.h"

namespace luminous {
    inline namespace gpu {

        MegakernelPT::MegakernelPT(Device *device, Context *context)
                : GPUIntegrator(device, context) {}

        void MegakernelPT::init_launch_params() {
            LaunchParams lp{};

            lp.sampler = _sampler.device_data();
            lp.camera = _camera.device_ptr();
            lp.frame_index = 0u;
            lp.max_depth = _max_depth;
            lp.min_depth = _min_depth;
            lp.rr_threshold = _rr_threshold;
            lp.scene_data = _scene->scene_data_device_ptr();
            _launch_params.reset(&lp);
        }

        void MegakernelPT::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            Integrator::init(scene_graph);
            _scene->init_accel<MegakernelOptixAccel>();
            init_on_device();
            init_launch_params();
        }

        void MegakernelPT::render(int frame_num, ProgressReporter *progressor) {
            auto res = _camera->resolution();
            for (int i = 0; i < frame_num; ++i) {
                _scene->accel<MegakernelOptixAccel>()->launch(res, _launch_params);
                if(progressor) progressor->update(1);
                _launch_params->frame_index += 1;
            }
        }

        void MegakernelPT::synchronize_to_gpu() {
            _camera.synchronize_all_to_device();
            _sampler.synchronize_to_device();
        }

        void MegakernelPT::update() {
            _launch_params->frame_index = 0u;
            synchronize_to_gpu();
        }

    }
}