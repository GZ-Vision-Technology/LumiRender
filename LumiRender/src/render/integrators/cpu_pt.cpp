//
// Created by Zero on 2021/5/29.
//

#include "cpu_pt.h"
#include "render/integrators/pt_func.h"
#include "render/scene/cpu_scene.h"
#include "render/sensors/common.h"

using std::cout;

using std::endl;

namespace luminous {
    inline namespace cpu {

        CPUPathTracer::CPUPathTracer(Device *device, Context *context)
                : Integrator(device, context) {
            set_thread_num(_context->thread_num());
        }

        void CPUPathTracer::init(const SP<SceneGraph> &scene_graph) {
            Integrator::init(scene_graph);
            _scene->init_accel<EmbreeAccel>();
        }

        void CPUPathTracer::update() {
            _frame_index = 0;
        }

        void CPUPathTracer::render() {
            const uint tile_size = 16;
            uint2 res = _camera->resolution();
            tiled_for_2d(res, make_uint2(tile_size), [&](uint2 pixel, uint tid) {
                Film *film = _camera->film();
                Sampler sampler = *_sampler.data();
                sampler.start_pixel_sample(pixel, _frame_index, 0);
                auto ss = sampler.sensor_sample(pixel);
                Ray ray{};
                float weight = _camera->generate_ray(ss, &ray);
                uint spp = sampler.spp();
                Spectrum L(0.f);
                for (int i = 0; i < spp; ++i) {
                    L += luminous::render::Li(ray, scene<CPUScene>()->scene_handle(), sampler,
                                              _max_depth, _rr_threshold, false,
                                              scene<CPUScene>()->scene_data());
                }
                L = L / float(spp);
                film->add_sample(pixel, L, weight, _frame_index);
            });
            ++_frame_index;
        }
    }
}