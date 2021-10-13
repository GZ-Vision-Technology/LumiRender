//
// Created by Zero on 2021/2/18.
//


#pragma once

#include <render/samplers/sampler.h>
#include "core/concepts.h"
#include "core/backend/device.h"
#include "util/parser.h"
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
            Managed<float4, float4> _render_buffer{_device.get()};
            Managed<float4, float4> _normal_buffer{_device.get()};
            Managed<float4, float4> _albedo_buffer{_device.get()};
            Managed<FrameBufferType, FrameBufferType> _frame_buffer{_device.get()};
        public:
            Task(std::unique_ptr<Device> device, Context *context)
            : _device(move(device)),
            _context(context) {}

            virtual void init(const Parser &parser);

            LM_NODISCARD std::shared_ptr<SceneGraph> build_scene_graph(const Parser &parser) {
                auto scene_graph = parser.parse();
                scene_graph->create_shapes();
                _output_config = scene_graph->output_config;
                return scene_graph;
            }

            virtual void update() {
                _integrator->update();
            }

            void save_to_file() {
                float4 *buffer = get_render_buffer();
                auto res = resolution();
                size_t size = res.x * res.y * pixel_size(PixelFormat::RGBA32F);
                auto p = new std::byte[size];
                Image image = Image(PixelFormat::RGBA32F, p, res);
                image.for_each_pixel([&](std::byte *pixel, int i) {
                    auto fp = reinterpret_cast<float4 *>(pixel);
                    *fp = Spectrum::linear_to_srgb(buffer[i]);
                });
                std::filesystem::path path = _context->scene_path() / _output_config.fn;
                image.save_image(path);
            }

            LM_NODISCARD bool complete() const {
                return _integrator->frame_index() >= _output_config.frame_num && _output_config.frame_num != 0;
            }

            virtual void render_gui(double dt) {
                _dt = dt;
                _integrator->render();

                if (_integrator->frame_index() == _output_config.frame_num
                && _output_config.frame_num != 0) {
                    save_to_file();
                }
            }

            virtual void update_device_buffer();

            LM_NODISCARD virtual Sensor *camera() {
                return _integrator->camera();
            }

            LM_NODISCARD virtual FrameBufferType *get_frame_buffer();

            LM_NODISCARD virtual float4 *get_render_buffer();

            LM_NODISCARD uint2 resolution();

            virtual void on_key(int key, int scancode, int action, int mods);

            void update_camera_fov_y(float val);

            virtual void update_camera_view(float d_yaw, float d_pitch);

            virtual void update_film_resolution(uint2 res);
        };
    }
}