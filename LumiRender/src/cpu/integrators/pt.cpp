//
// Created by Zero on 2021/5/29.
//

#include "pt.h"
#include "cpu/cpu_scene.h"
#include "render/integrators/pt_func.h"
#include "util/parallel.h"

using std::cout;

using std::endl;

namespace luminous {
    inline namespace cpu {

        CPUPathTracer::CPUPathTracer(Context *context)
                : _context(context) {
            set_thread_num(_context->thread_num());
        }

        void CPUPathTracer::init(const SP<SceneGraph> &scene_graph) {
            _scene = std::make_unique<CPUScene>(_context);
            init_with_config(scene_graph->integrator_config);
            _scene->init(scene_graph);
            _camera = Sensor::create(scene_graph->sensor_config);
            _sampler = Sampler::create(scene_graph->sampler_config);

        }

        Sensor *CPUPathTracer::camera() {
            return &_camera;
        }

        void CPUPathTracer::update() {
            _frame_index = 0;
        }

        void CPUPathTracer::render() {
            const uint tile_size = 16;
            uint2 res = _camera.resolution();
            uint2 n_tiles = (res + tile_size - 1u) / tile_size;
            parallel_for_2d(n_tiles, [&](uint2 tile, uint thread_id) {
                uint2 p_min = tile * tile_size;
                uint2 p_max = p_min + tile_size;
                p_max = select(p_max > res, res, p_max);
                Box2u tile_bound{p_min, p_max};
                Sampler sampler = _sampler;
                tile_bound.for_each([&](uint2 pixel) {
                    Film *film = _camera.film();
                    sampler.start_pixel_sample(pixel, _frame_index, 0);
                    auto ss = sampler.sensor_sample(pixel);
                    Ray ray{};
                    float weight = _camera.generate_ray(ss, &ray);
                    uint spp = sampler.spp();
                    Spectrum L(0.f);
                    for (int i = 0; i < spp; ++i) {
                        L += Li(ray, (uint64_t)_scene->rtc_scene(), sampler,
                                _max_depth, _rr_threshold, false, _scene->scene_data());
                    }
                    L = L / float(spp);
                    film->add_sample(pixel, L, weight, _frame_index);
                });
            });
            ++_frame_index;
        }
    }
}