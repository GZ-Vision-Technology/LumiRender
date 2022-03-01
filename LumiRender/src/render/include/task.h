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
#include "util/progressreporter.h"

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
            FBState _fb_state;
            Managed<float4, float4> _render_buffer{_device.get()};
            Managed<float4, float4> _normal_buffer{_device.get()};
            Managed<float4, float4> _albedo_buffer{_device.get()};
            Managed<FrameBufferType, FrameBufferType> _frame_buffer{_device.get()};
            ProgressReporter _progressor;
        public:
            Task(std::unique_ptr<Device> device, Context *context)
                    : _device(move(device)),
                      _context(context) {}

            void init(const Parser &parser);
            void post_init();

            LM_NODISCARD std::shared_ptr<SceneGraph> build_scene_graph(const Parser &parser);

            void update() {
                _dispatch_num = 0;
                _integrator->update();
            }

            void finalize();

            float get_fps() const;

            void save_to_file();

            LM_NODISCARD bool complete() const {
                return _dispatch_num >= _output_config.dispatch_num;
            }

            bool result_available() const {
                return _dispatch_num > 0;
            }

            void render_gui(double dt);

            void update_device_buffer();

            LM_NODISCARD Sensor *camera() { return _integrator->camera(); }

            LM_NODISCARD FrameBufferType *get_frame_buffer(bool host_side = true);

            LM_NODISCARD float4 *get_render_buffer(bool host_side = true);

            LM_NODISCARD float4 *get_normal_buffer(bool host_side = true);

            LM_NODISCARD float4 *get_albedo_buffer(bool host_side = true);

            LM_NODISCARD float4 *get_buffer(bool host_side = true);

            LM_NODISCARD uint2 resolution();

            void on_key(int key, int scancode, int action, int mods);

            void update_camera_fov_y(float val);

            void update_camera_view(float d_yaw, float d_pitch);

            void update_film_resolution(uint2 res);
        };
    }
}