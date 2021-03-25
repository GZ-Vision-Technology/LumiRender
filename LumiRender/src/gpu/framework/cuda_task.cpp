//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"

namespace luminous {
    inline namespace gpu {

        void CUDATask::on_key(int key, int scancode, int action, int mods) {
            float3 forward = _camera.forward();
            float3 up = _camera.up();
            float3 right = _camera.right();
            float velocity = _camera.velocity();
            switch (key) {
                case 'A':
                    _camera.move(-right * velocity);
                    break;
                case 'S':
                    _camera.move(-forward * velocity);
                    break;
                case 'D':
                    _camera.move(right * velocity);
                    break;
                case 'W':
                    _camera.move(forward * velocity);
                    break;
                case 'Q':
                    _camera.move(-up * velocity);
                    break;
                case 'E':
                    _camera.move(up * velocity);
                    break;
                default:
                    break;
            }
        }

        void CUDATask::update_camera_fov_y(float val) {
            _camera.update_fov_y(val);
        }

        void CUDATask::update_camera_view(float d_yaw, float d_pitch) {
            _camera.update_yaw(d_yaw);
            _camera.update_pitch(d_pitch);
        }

        void CUDATask::update_film_resolution(int2 res) {
            auto film = _camera.film();
            film->set_resolution(res);
            update_device_buffer();
        }

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _camera = SensorHandle::create(scene_graph->sensor_config);
            _integrator = make_unique<MegaKernelPT>(_device);
            _integrator->init(scene_graph, &_camera);
            update_device_buffer();
        }

        void CUDATask::update_device_buffer() {
            auto res = _camera.film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer = _device->allocate_buffer<float4>(num);
            _camera.film()->set_accumulate_buffer(_accumulate_buffer.data());

            _host_frame_buffer.resize(num);

            _frame_buffer.reset(_host_frame_buffer.data(), _device, num);
            _frame_buffer.synchronize_to_gpu();
            _camera.film()->set_frame_buffer(_frame_buffer.device_data());
        }

        void CUDATask::render_gui() {
            _integrator->update();
            _integrator->render();
        }

        int2 CUDATask::resolution() {
            return _camera.film()->resolution();
        }

        FrameBufferType *CUDATask::download_frame_buffer() {
            _frame_buffer.synchronize_to_cpu();
            return _frame_buffer.get();
        }
    }
}