//
// Created by Zero on 2021/3/23.
//


#pragma once

#include "render/include/integrator.h"

#include "framework/optix_accel.h"
#include "gpu_scene.h"
#include "render/samplers/sampler.h"

namespace luminous {

    inline namespace gpu {
        class MegaKernelPT : public Integrator {
        private:
            Managed<Sampler> _sampler;
            Managed<Sensor> _camera;
            UP<GPUScene> _scene{nullptr};
            SP<Device> _device{};
            Managed<LaunchParams> _launch_params;
        public:

            MegaKernelPT(const SP<Device> &device)
                    : _device(device) {}

            void init(const SP<SceneGraph> &scene_graph) override;

            void init_launch_params();

            Sensor *camera() override;

            void update() override;

            void synchronize_to_gpu();

            void render() override;

            void update_camera_fov_y(float val);

            void update_camera_view(float d_yaw, float d_pitch);

            void update_film_resolution(uint2 res);
        };
    }
}