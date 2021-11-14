//
// Created by Zero on 12/10/2021.
//

#include "task.h"
#include "render/integrators/megakernel_pt.h"
#include "render/integrators/cpu_pt.h"
#include "render/integrators/wavefront/integrator.h"

namespace luminous {
    inline namespace render {
        void Task::on_key(int key, int scancode, int action, int mods) {
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

        void Task::update_camera_view(float d_yaw, float d_pitch) {
            float sensitivity = camera()->sensitivity();
            camera()->update_yaw(d_yaw * sensitivity);
            camera()->update_pitch(d_pitch * sensitivity);
        }

        uint2 Task::resolution() {
            return camera()->film()->resolution();
        }

        void Task::update_camera_fov_y(float val) {
            camera()->update_fov_y(val);
        }

        void Task::update_film_resolution(uint2 res) {
            camera()->update_film_resolution(res);
            update_device_buffer();
        }

        void Task::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;

            _render_buffer.resize(num, make_float4(0.f));
            _render_buffer.allocate_device(num);
            camera()->film()->set_render_buffer_view(_render_buffer.device_buffer_view());

            _normal_buffer.resize(num, make_float4(0.f));
            _normal_buffer.allocate_device(num);
            camera()->film()->set_normal_buffer_view(_normal_buffer.device_buffer_view());

            _albedo_buffer.resize(num, make_float4(0.f));
            _albedo_buffer.allocate_device(num);
            camera()->film()->set_albedo_buffer_view(_albedo_buffer.device_buffer_view());

            _frame_buffer.reset(num);
            _frame_buffer.synchronize_to_device();
            camera()->film()->set_frame_buffer_view(_frame_buffer.device_buffer_view());

        }

        FrameBufferType *Task::get_frame_buffer() {
            return _frame_buffer.synchronize_and_get_host_data();;
        }

        float4 *Task::get_render_buffer() {
            return _render_buffer.synchronize_and_get_host_data();
        }

        void Task::init(const JsonParser &parser) {
            auto scene_graph = build_scene_graph(parser);
            const std::string type = scene_graph->integrator_config.type();
            if (type == "PT") {
                if (_device->is_cpu()) {
                    _integrator = std::make_unique<CPUPathTracer>(_device.get(), _context);
                } else {
                    _integrator = std::make_unique<MegakernelPT>(_device.get(), _context);
                }
            } else if(type == "WavefrontPT") {
                _integrator = std::make_unique<WavefrontPT>(_device.get(), _context);
            }
            _integrator->init(scene_graph);
            update_device_buffer();
        }
    }
}