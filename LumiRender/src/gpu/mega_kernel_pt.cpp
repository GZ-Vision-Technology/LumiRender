//
// Created by Zero on 2021/3/23.
//

#include "mega_kernel_pt.h"

namespace luminous {
    inline namespace gpu {

        void MegaKernelPT::init_launch_params() {
            LaunchParams lp{};
            lp.sampler = _sampler.device_data();
            lp.camera = _camera.device_data();
            lp.frame_index = 0u;
            _launch_params.reset(&lp, _device);
            _launch_params.synchronize_to_gpu();
        }

        void MegaKernelPT::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            _scene = make_unique<GPUScene>(_device);
            _scene->init(scene_graph);
            auto camera = Sensor::create(scene_graph->sensor_config);
            _camera.reset(&camera, _device);
            auto sampler = SamplerHandle::create(scene_graph->sampler_config);
            _sampler.reset(&sampler, _device);
            init_launch_params();
        }

        void MegaKernelPT::update_camera_fov_y(float val) {
            _camera->update_fov_y(val);
        }

        void MegaKernelPT::update_camera_view(float d_yaw, float d_pitch) {
            _camera->update_yaw(d_yaw);
            _camera->update_pitch(d_pitch);
        }

        void MegaKernelPT::update_film_resolution(uint2 res) {
            auto film = _camera->film();
            film->set_resolution(res);
        }

        void MegaKernelPT::render() {
            auto res = _camera->resolution();
            _launch_params->frame_index += 1;
            _scene->launch(res, _launch_params);
        }

        void MegaKernelPT::synchronize_to_gpu() {
            _camera.synchronize_to_gpu();
            _sampler.synchronize_to_gpu();
        }

        void MegaKernelPT::update() {
            _launch_params->frame_index = 0u;
            synchronize_to_gpu();
        }

        Sensor *MegaKernelPT::camera() {
            return _camera.get();
        }
    }
}