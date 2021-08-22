//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/include/integrator.h"
#include "gpu/framework/megakernel_optix_accel.h"
#include "render/samplers/sampler.h"

namespace luminous {

    inline namespace gpu {
        class GPUScene;

        class MegakernelPT : public Integrator {
        private:
            Context *_context{};
            Managed<Sampler, Sampler> _sampler;
            Managed<Sensor, Sensor> _camera;
            UP<GPUScene> _scene{nullptr};
            SP<Device> _device{};
            Managed<LaunchParams> _launch_params;
        public:

            MegakernelPT(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void init_launch_params();

            NDSC uint frame_index() const override {
                return _launch_params->frame_index;
            }

            Sensor *camera() override;

            void update() override;

            void synchronize_to_gpu();

            void render() override;
        };
    }
}