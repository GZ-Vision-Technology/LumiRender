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
        using std::shared_ptr;

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
            shared_ptr<Kernel> _generate_ray{};
            shared_ptr<Kernel> _queues_reset{};
            shared_ptr<Kernel> _generate_ray_samples{};
            shared_ptr<Kernel> _process_escape_ray{};
            shared_ptr<Kernel> _process_emission{};
            shared_ptr<Kernel> _eval_BSDFs{};
        public:

            WavefrontPT(Device *device, Context *context);

            void init(const shared_ptr<SceneGraph> &scene_graph) override;

            void allocate_memory();

            NDSC uint frame_index() const override { return _frame_index; }

            void update() override {

            }

            RayQueue *current_ray_queue(int wavefrontDepth) {
                return _ray_queues.device_buffer().address<RayQueue *>(wavefrontDepth & 1);
            }

            RayQueue *next_ray_queue(int wavefrontDepth) {
                return _ray_queues.device_buffer().address<RayQueue *>((wavefrontDepth + 1) & 1);
            }

            void render() override;

        };
    }
}