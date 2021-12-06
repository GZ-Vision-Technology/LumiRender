//
// Created by Zero on 2021/2/18.
//


#pragma once

#include <render/samplers/sampler.h>
#include "core/concepts.h"
#include "core/backend/device.h"
#include "parser/json_parser.h"
#include "render/integrators/integrator.h"
#include "core/backend/managed.h"
#include "render/sensors/common.h"

namespace luminous {
    inline namespace render {
        using std::unique_ptr;

        class Sensor;

        class Task : public Noncopyable {
        protected:
            std::unique_ptr<Device> _device{nullptr};
            Context *_context{nullptr};
            UP<Integrator> _integrator;
            double _dt{0};
            OutputConfig _output_config;
            int _dispatch_num{0};
            Managed<float4, float4> _render_buffer{_device.get()};
            Managed<float4, float4> _normal_buffer{_device.get()};
            Managed<float4, float4> _albedo_buffer{_device.get()};
            Managed<FrameBufferType, FrameBufferType> _frame_buffer{_device.get()};
        public:
            Task(std::unique_ptr<Device> device, Context *context)
                    : _device(move(device)),
                      _context(context) {}

            void init(const Parser &parser);

            LM_NODISCARD std::shared_ptr<SceneGraph> build_scene_graph(const Parser &parser);

            void update() { _integrator->update(); }

            void save_to_file();

            LM_NODISCARD bool complete() const {
                return _dispatch_num >= _output_config.dispatch_num;
            }

            void render_gui(double dt);

            void update_device_buffer();

            LM_NODISCARD virtual Sensor *camera() { return _integrator->camera(); }

            LM_NODISCARD virtual FrameBufferType *get_frame_buffer();

            LM_NODISCARD virtual float4 *get_render_buffer();

            LM_NODISCARD uint2 resolution();

            void on_key(int key, int scancode, int action, int mods);

            void update_camera_fov_y(float val);

            void update_camera_view(float d_yaw, float d_pitch);

            void update_film_resolution(uint2 res);
        };
    }
}