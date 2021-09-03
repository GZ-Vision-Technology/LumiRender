//
// Created by Zero on 2021/3/24.
//

#include "integrator.h"

namespace luminous {
    inline namespace gpu {

        WavefrontPT::WavefrontPT(const SP<Device> &device, Context *context)
                : Integrator(device, context) {

        }

        void WavefrontPT::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            Integrator::init(scene_graph);
            // todo: make this configurable. Base it on the amount of GPU memory?
            int max_samples = 1024 * 1024;
            uint2 res = resolution();
            _scanline_per_pass = std::max(1, int(max_samples / res.x));
            int n_passes = (res.y + _scanline_per_pass - 1) / _scanline_per_pass;
            _max_queue_size = res.x * _scanline_per_pass;
        }

        void WavefrontPT::render() {

        }

    }
}