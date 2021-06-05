//
// Created by Zero on 2021/5/29.
//

#include "pt.h"
#include "cpu/cpu_scene.h"

namespace luminous {
    inline namespace cpu {

        CPUPathTracer::CPUPathTracer(Context *context) {

        }

        void CPUPathTracer::init(const SP<SceneGraph> &scene_graph) {

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