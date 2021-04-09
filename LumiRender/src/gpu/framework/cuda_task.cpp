//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"

namespace luminous {
    inline namespace gpu {

        void CUDATask::on_key(int key, int scancode, int action, int mods) {
            auto p_camera = camera();
            float3 forward = p_camera->forward();
            float3 up = p_camera->up();
            float3 right = p_camera->right();
            float distance = p_camera->velocity() * _dt;
            switch (key) {
                case 'A':
                    p_camera->move(-right * distance);
                    break;
                case 'S':
                    p_camera->move(-forward * distance);
                    break;
                case 'D':
                    p_camera->move(right * distance);
                    break;
                case 'W':
                    p_camera->move(forward * distance);
                    break;
                case 'Q':
                    p_camera->move(-up * distance);
                    break;
                case 'E':
                    p_camera->move(up * distance);
                    break;
                default:
                    break;
            }
        }

        void CUDATask::update_camera_fov_y(float val) {
            camera()->update_fov_y(val);
        }

        void CUDATask::update_camera_view(float d_yaw, float d_pitch) {
            float sensitivity = camera()->sensitivity();
            camera()->update_yaw(d_yaw * sensitivity);
            camera()->update_pitch(d_pitch * sensitivity);
        }

        void CUDATask::update_film_resolution(uint2 res) {
            camera()->update_film_resolution(res);
            update_device_buffer();
        }

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _integrator = make_unique<MegaKernelPT>(_device);
            _integrator->init(scene_graph);
            update_device_buffer();
        }

        void CUDATask::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer = _device->allocate_buffer<float4>(num);
            camera()->film()->set_accumulate_buffer(_accumulate_buffer.data());

            _frame_buffer.reset(_device, num);
            _frame_buffer.synchronize_to_gpu();
            camera()->film()->set_frame_buffer(_frame_buffer.device_data());
        }

        void CUDATask::update() {
            _integrator->update();
        }

        void CUDATask::render_gui(double dt) {
            _dt = dt;
//            _integrator->update();
            _integrator->render();
        }

        uint2 CUDATask::resolution() {
            return camera()->film()->resolution();
        }

        FrameBufferType *CUDATask::download_frame_buffer() {
            _frame_buffer.synchronize_to_cpu();
            return _frame_buffer.get();
        }

        Sensor *CUDATask::camera() {
            return _integrator->camera();
        }

    }
}