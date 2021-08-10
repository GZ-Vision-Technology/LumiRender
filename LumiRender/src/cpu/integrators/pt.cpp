//
// Created by Zero on 2021/5/29.
//

#include "pt.h"
#include "cpu/cpu_scene.h"
#include "render/integrators/pt_func.h"
#include "util/parallel.h"

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

        }

        void CPUPathTracer::render() {
            const uint tile_size = 16u;
            uint2 res = _camera.resolution();
            uint2 n_tiles = (res + tile_size - 1u) / tile_size;

            parallel_for_2d(n_tiles, [&](uint2 pixel, uint thread_id) {
                pixel.print();
            });
        }
    }
}