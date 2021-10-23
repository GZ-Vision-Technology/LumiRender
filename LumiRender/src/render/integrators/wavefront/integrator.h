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
#include "render/include/kernel.h"
#include "aggregate.h"
#include "kernels.h"

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
            Managed<RTParam> _rt_param{_device};

#define DEFINE_KERNEL(arg) Kernel<decltype(&(arg))> _##arg{arg};
            // kernels
            DEFINE_KERNEL(generate_primary_ray);
            DEFINE_KERNEL(generate_ray_samples);
            DEFINE_KERNEL(process_escape_ray);
            DEFINE_KERNEL(process_emission);
            DEFINE_KERNEL(eval_BSDFs);
#undef DEFINE_KERNEL

            // accelerate structure
            WavefrontAggregate *_aggregate{};

            std::shared_ptr<Module> _module;
        private:
            LM_NODISCARD static int _cur_index(int depth) { return depth & 1; }

            LM_NODISCARD static int _next_index(int depth) { return (depth + 1) & 1; }

            LM_NODISCARD RayQueue *_current_ray_queue(int depth) {
                return _ray_queues.device_buffer().address<RayQueue *>(_cur_index(depth));
            }

            LM_NODISCARD RayQueue *_next_ray_queue(int depth) {
                return _ray_queues.device_buffer().address<RayQueue *>(_next_index(depth));
            }

            void _reset_cur_ray_queue(int depth) {
                RayQueue *current_ray_queue = _current_ray_queue(depth);
                current_ray_queue->reset();
                _ray_queues.synchronize_to_device(_cur_index(depth), 1);
            }

            void _reset_next_ray_queue(int depth) {
                RayQueue *next_ray_queue = _next_ray_queue(depth);
                next_ray_queue->reset();
                _ray_queues.synchronize_to_device(_next_index(depth), 1);
            }
        public:

            WavefrontPT(Device *device, Context *context);

            void init(const shared_ptr<SceneGraph> &scene_graph) override;

            void reset_queues(int depth);

            void init_aggregate();

            void init_kernels();

            void init_rt_param();

            void load_module();

            void allocate_memory();

            LM_NODISCARD uint frame_index() const override { return _rt_param->frame_index; }

            void intersect_closest(int wavefront_depth);

            void trace_shadow_ray(int wavefront_depth);

            void update() override {
                _rt_param->frame_index += 1;
                _rt_param.synchronize_to_device();
                _camera.synchronize_all_to_device();
            }

            void render() override;

            void render_per_sample(int sample_idx);

        };
    }
}