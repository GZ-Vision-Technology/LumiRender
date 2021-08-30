//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/integrators/integrator.h"
#include "gpu/accel/megakernel_optix_accel.h"
#include "render/samplers/sampler.h"
#include "render/include/scene.h"

namespace luminous {

    inline namespace gpu {

        class MegakernelPT : public Integrator {
        private:
            Managed<Sampler, Sampler> _sampler;
            Managed<Sensor, Sensor> _camera;
            Managed<LaunchParams> _launch_params;
        public:

            MegakernelPT(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void init_launch_params();

            NDSC uint frame_index() const override {
                return _launch_params->frame_index;
            }

            template<typename TScene>
            NDSC decltype(auto) scene() {
                return reinterpret_cast<TScene*>(_scene.get());
            }

            Sensor *camera() override;

            void update() override;

            void synchronize_to_gpu();

            void render() override;
        };
    }
}