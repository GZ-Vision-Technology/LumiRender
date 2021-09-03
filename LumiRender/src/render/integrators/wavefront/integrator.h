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
#include "params.h"

namespace luminous {
    inline namespace gpu {
        class WavefrontPT : public Integrator {
        private:
            RayQueue *_ray_queues[2]{};
            ShadowRayQueue * _shadow_ray_queue{};
            HitAreaLightQueue * _hit_area_light_queue{};
            EscapedRayQueue * _escaped_ray_queue{};
            MaterialEvalQueue *_material_eval_queue{};
            SOA<PixelSampleState> * _pixel_sample_state{};
            int _scanline_per_pass{};
            int _max_queue_size{};
        public:

            WavefrontPT(const SP<Device> &device, Context *context);

            void init(const std::shared_ptr<SceneGraph> &scene_graph) override;

            NDSC uint frame_index() const override {
                return 0;
            }

            void update() override {

            }

            RayQueue *current_ray_queue(int wavefrontDepth) {
                return _ray_queues[wavefrontDepth & 1];
            }

            RayQueue *next_ray_queue(int wavefrontDepth) {
                return _ray_queues[(wavefrontDepth + 1) & 1];
            }

            void render() override;

        };
    }
}