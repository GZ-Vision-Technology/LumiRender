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
#include "parser/scene_graph.h"

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
            std::shared_ptr<SceneGraph> _scene_graph{};
            int _spp{0};
            FBState _fb_state{};
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

            void update() {
                _spp = 0;
                _integrator->update();
            }

            void finalize();

            float get_fps() const;

            void save_to_file();

            void update_sensor(const SensorConfig &config);

            LM_NODISCARD bool complete() const {
                return _spp >= _scene_graph->output_config.spp;
            }

            bool result_available() const {
                return _spp > 0;
            }

            void run();

            void save_render_result(const std::string &fn);

            void render(double dt);

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