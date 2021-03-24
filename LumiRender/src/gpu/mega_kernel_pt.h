//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/include/integrator.h"

#include "framework/optix_accel.h"
#include "gpu_scene.h"
#include "render/samplers/sampler_handle.h"

namespace luminous {

    inline namespace gpu {
        class MegaKernelPT : public Integrator {
        private:
            Managed<SamplerHandle> _sampler;
            Managed<const SensorHandle *> _camera;
            UP<GPUScene> _scene{nullptr};
            SP<Device> _device{};
        public:

            MegaKernelPT(const SP<Device> &device)
                    : _device(device) {}

            void init(const SP<SceneGraph> &scene_graph, SensorHandle *camera) override;

            void render() override;
        };
    }
}