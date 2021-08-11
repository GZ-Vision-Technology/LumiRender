//
// Created by Zero on 2021/2/18.
//


#pragma once

#include <render/samplers/sampler.h>
#include "core/concepts.h"
#include "device.h"
#include "render/include/parser.h"
#include "render/sensors/sensor.h"
#include "render/include/integrator.h"
#include "core/backend/managed.h"

namespace luminous {
    using std::unique_ptr;
    class Task : public Noncopyable {
    protected:
        std::shared_ptr<Device> _device{nullptr};
        Context * _context{nullptr};
        UP<Integrator> _integrator;
        double _dt{0};
        Managed<float4, float4> _accumulate_buffer;
        Managed<FrameBufferType, FrameBufferType> _frame_buffer;
    public:
        Task(const std::shared_ptr<Device> &device, Context *context)
            : _device(device),
            _context(context) {}

        virtual void init(const Parser &parser) = 0;

        virtual void update() {
            _integrator->update();
        }

        virtual void render_gui(double dt) = 0;

        virtual void render_cli() = 0;

        virtual void update_device_buffer() = 0;

        virtual Sensor * camera() {
            return _integrator->camera();
        }

        virtual FrameBufferType *download_frame_buffer() = 0;

        NDSC uint2 resolution() {
            return camera()->film()->resolution();
        }

        virtual void on_key(int key,int scancode, int action, int mods) {
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

        void update_camera_fov_y(float val) {
            camera()->update_fov_y(val);
        }

        virtual void update_camera_view(float d_yaw, float d_pitch) {
            float sensitivity = camera()->sensitivity();
            camera()->update_yaw(d_yaw * sensitivity);
            camera()->update_pitch(d_pitch * sensitivity);
        }

        virtual void update_film_resolution(uint2 res) {
            camera()->update_film_resolution(res);
            update_device_buffer();
        }
    };
}