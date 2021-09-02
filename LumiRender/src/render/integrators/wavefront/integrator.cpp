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
            init_with_config(scene_graph->integrator_config);

        }

        void WavefrontPT::render() {

        }

    }
}