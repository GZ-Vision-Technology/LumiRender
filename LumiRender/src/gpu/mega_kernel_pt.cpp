//
// Created by Zero on 2021/3/23.
//

#include "mega_kernel_pt.h"

namespace luminous {
    inline namespace gpu {

        void MegaKernelPT::init(const std::shared_ptr<SceneGraph> &scene_graph,
                                SensorHandle *camera) {
            _scene = make_unique<GPUScene>(_device);
            _scene->init(scene_graph);

            _camera.reset(camera, _device);
        }

        void MegaKernelPT::render() {

        }
    }
}