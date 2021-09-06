//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "gpu/integrators/integrator.h"
#include "render/sensors/camera_base.h"
#include "gpu/accel/megakernel_optix_accel.h"
#include "gpu/gpu_scene.h"
#include "render/samplers/sampler.h"
#include "work_items.h"
#include "core/backend/device.h"
#include "core/backend/kernel.h"

namespace luminous {
    inline namespace gpu {
        class WavefrontPT : public Integrator {
        private:
            // queue
            Managed<RayQueue, RayQueue> _ray_queues;
            Managed<ShadowRayQueue, ShadowRayQueue> _shadow_ray_queue;
            Managed<HitAreaLightQueue, HitAreaLightQueue> _hit_area_light_queue;
            Managed<EscapedRayQueue, EscapedRayQueue> _escaped_ray_queue;
            Managed<MaterialEvalQueue, MaterialEvalQueue> _material_eval_queue;
            Managed<SOA<PixelSampleState>, SOA<PixelSampleState>> _pixel_sample_state;

            // base params
            uint _scanline_per_pass{};
            uint _max_queue_size{};
            uint _frame_index{};

            // kernels
            std::shared_ptr<Kernel> _raygen_kernel;

        public:

            WavefrontPT(Device *device, Context *context);

            void init(const std::shared_ptr<SceneGraph> &scene_graph) override;

            void allocate_memory();

            NDSC uint frame_index() const override { return _frame_index; }

            void update() override {

            }

            RayQueue &current_ray_queue(int wavefrontDepth) {
                return _ray_queues[wavefrontDepth & 1];
            }

            RayQueue &next_ray_queue(int wavefrontDepth) {
                return _ray_queues[(wavefrontDepth + 1) & 1];
            }

            void render() override;

        };
    }
}