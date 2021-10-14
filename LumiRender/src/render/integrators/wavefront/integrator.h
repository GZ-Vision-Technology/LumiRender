//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "render/integrators/integrator.h"
#include "render/sensors/camera_base.h"
#include "render/scene/gpu_scene.h"
#include "render/samplers/sampler.h"
#include "work_items.h"
#include "core/backend/device.h"
#include "core/backend/kernel.h"
#include "aggregate.h"

namespace luminous {
    inline namespace gpu {
        using std::shared_ptr;

        class WavefrontPT : public Integrator {
        private:
            // queue
            Managed<RayQueue, RayQueue> _ray_queues{_device};
            Managed<ShadowRayQueue, ShadowRayQueue> _shadow_ray_queue{_device};
            Managed<HitAreaLightQueue, HitAreaLightQueue> _hit_area_light_queue{_device};
            Managed<EscapedRayQueue, EscapedRayQueue> _escaped_ray_queue{_device};
            Managed<MaterialEvalQueue, MaterialEvalQueue> _material_eval_queue{_device};
            Managed<SOA<PixelSampleState>, SOA<PixelSampleState>> _pixel_sample_state{_device};

            // base params
            int _scanline_per_pass{};
            int _max_queue_size{};
            int _frame_index{};

            // kernels
            shared_ptr<Kernel> _generate_ray{};
            shared_ptr<Kernel> _queues_reset{};
            shared_ptr<Kernel> _generate_ray_samples{};
            shared_ptr<Kernel> _process_escape_ray{};
            shared_ptr<Kernel> _process_emission{};
            shared_ptr<Kernel> _eval_BSDFs{};

            // accelerate structure
            WavefrontAggregate *_aggregate{};
        public:

            WavefrontPT(Device *device, Context *context);

            void init(const shared_ptr<SceneGraph> &scene_graph) override;

            void init_aggregate();

            void allocate_memory();

            LM_NODISCARD uint frame_index() const override { return _frame_index; }

            void intersect_closest(int wavefront_depth);

            void trace_shadow_ray(int wavefront_depth);

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