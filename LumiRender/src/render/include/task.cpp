//
// Created by Zero on 12/10/2021.
//

#include "task.h"
#include "render/integrators/megakernel_pt.h"
#include "render/integrators/cpu_pt.h"
#include "render/integrators/wavefront/integrator.h"
#include "denoise/denoiser.h"

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

        FrameBufferType *Task::get_frame_buffer(bool host_side) {
            if (host_side)
                return _frame_buffer.synchronize_and_get_host_data();
            else
                return _frame_buffer.device_data();
        }

        float4 *Task::get_render_buffer(bool host_side) {
            if (host_side)
                return _render_buffer.synchronize_and_get_host_data();
            else
                return _render_buffer.device_data();
        }

        float4 *Task::get_normal_buffer(bool host_side) {
            if (host_side)
                return _normal_buffer.synchronize_and_get_host_data();
            else
                return _normal_buffer.device_data();
        }

        float4 *Task::get_albedo_buffer(bool host_side) {
            if (host_side)
                return _albedo_buffer.synchronize_and_get_host_data();
            else
                return _albedo_buffer.device_data();
        }

        float4 *Task::get_buffer(bool host_side) {
            auto fb_state = camera()->film()->frame_buffer_state();
            switch (fb_state) {
                case Render:
                    return get_render_buffer(host_side);
                case Normal:
                    return get_normal_buffer(host_side);
                case Albedo:
                    return get_albedo_buffer(host_side);
            }
            DCHECK(0);
            return nullptr;
        }

        void Task::init(const Parser &parser) {
            auto scene_graph = build_scene_graph(parser);
            const std::string type = scene_graph->integrator_config.type();
            if (type == "PT") {
                if (_device->is_cpu()) {
                    _integrator = std::make_unique<CPUPathTracer>(_device.get(), _context);
                } else {
                    _integrator = std::make_unique<MegakernelPT>(_device.get(), _context);
                }
            } else if (type == "WavefrontPT") {
                _integrator = std::make_unique<WavefrontPT>(_device.get(), _context);
            }
            _integrator->init(scene_graph);
            update_device_buffer();
            _clock.reset();
        }

        std::shared_ptr<SceneGraph> Task::build_scene_graph(const Parser &parser) {
            auto scene_graph = parser.parse();
            _output_config = scene_graph->output_config;
            return scene_graph;
        }

        void Task::save_to_file(const OutputConfig &oc) {
            float4 *buffer = get_buffer();
            auto res = resolution();
            size_t size = res.x * res.y * pixel_size(PixelFormat::RGBA32F);
            Image image = Image::from_data(buffer, res);
            image.for_each_pixel([&](std::byte *pixel, int i) {
                auto fp = reinterpret_cast<float4 *>(pixel);
                float4 val = buffer[i];
                *fp = Spectrum::tone_mapping(val, oc.tone_map);
            });

            auto change_fn = [&](const std::filesystem::path &output_path,
                                 const std::string &suffix,
                                 std::string ext = "") {
                ext = ext.empty() ? output_path.extension().string() : ext;
                auto fn = output_path.stem().string() + suffix + ext;
                return output_path.parent_path() / fn;
            };

            luminous_fs::path film_out_path = _context->output_film_path();
            luminous_fs::path scene_path = _context->scene_file();
            if (film_out_path.empty()) {
                // if command line --output is empty, retrieve path from scene description file.
                film_out_path = _output_config.fn;
            }

            if (film_out_path.empty() || !film_out_path.has_filename()) {
                // default path is <scene path without extend>.exr
                film_out_path = scene_path.parent_path() / scene_path.stem();
                film_out_path += ".exr";
            } else if (film_out_path.is_relative()) {
                // Relative path is relative to scene directory.
                film_out_path = scene_path.parent_path() / film_out_path;
            }
            float4 *render = _render_buffer.synchronize_and_get_host_data();
            float4 *normal = _normal_buffer.synchronize_and_get_host_data();
            float4 *albedo = _albedo_buffer.synchronize_and_get_host_data();
            // denoising
            if (_context->denoise_output()) {
                auto denoiser = Denoiser();
                auto image_denoised = Image::create_empty(PixelFormat::RGBA32F, res);

                denoiser.execute(res, image_denoised.pixel_ptr<float4>(), render, normal, albedo);
                image_denoised.for_each_pixel([&](std::byte *pixel, int i) {
                    auto fp = reinterpret_cast<float4 *>(pixel);
                    float4 val = *fp;
                    val.w = 1.f;
                    *fp = Spectrum::tone_mapping(val, oc.tone_map);
                });
                auto denoised_fn = change_fn(film_out_path,"-denoised");
                image_denoised.save(denoised_fn);
            }
            if (oc.albedo) {
                auto image_albedo = Image::create_empty(PixelFormat::RGBA32F, res);
                image_albedo.for_each_pixel([&](std::byte *pixel, int i) {
                    auto fp = reinterpret_cast<float4 *>(pixel);
                    *fp = albedo[i];
                });
                image_albedo.save(change_fn(film_out_path, "-albedo"));
            }
            if (oc.normal_remapping) {
                auto image_normal = Image::create_empty(PixelFormat::RGBA32F, res);
                image_normal.for_each_pixel([&](std::byte *pixel, int i) {
                    auto fp = reinterpret_cast<float4 *>(pixel);
                    *fp = (normal[i] + 1.f) / 2.f;
                });
                image_normal.save(change_fn(film_out_path, "-normal_remapping"));
            }
            // todo hdr save bug
            image.save(film_out_path);
        }

        void Task::render_gui(double dt) {
            _dt = dt;
            _integrator->render(_output_config.frame_per_dispatch);
            if (_dispatch_num++ == _output_config.dispatch_num
                && _output_config.dispatch_num != 0) {
                auto t = _clock.get_time();
                cout << "render is complete elapse " << t << " s" << endl;
                cout << (_output_config.dispatch_num * _output_config.frame_per_dispatch) / t << " fps" << endl;
                save_to_file(_output_config);
            }
        }
    }
}