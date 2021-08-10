//
// Created by Zero on 2021/5/29.
//

#include "pt.h"
#include "cpu/cpu_scene.h"
#include "render/integrators/pt_func.h"
#include ""

namespace luminous {
    inline namespace cpu {

        CPUPathTracer::CPUPathTracer(Context *context) {

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

        }
    }
}