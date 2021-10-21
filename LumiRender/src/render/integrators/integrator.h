//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "render/scene/scene_graph.h"
#include "core/backend/synchronizer.h"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"


namespace luminous {
    inline namespace render {

        class Scene;

        class Integrator : public Noncopyable {
        protected:
            uint _max_depth{};
            float _rr_threshold{};
            Device *_device{};
            SP<Scene> _scene{};
            Context *_context{};
            Managed<Sampler, Sampler> _sampler{_device};
            Synchronizer<Sensor> _camera{_device};
            Dispatcher _dispatcher;
        public:
            Integrator(Device *device, Context *context)
                    : _device(device),
                      _context(context),
                      _dispatcher(_device->new_dispatcher()) {}

            virtual ~Integrator() = default;

            virtual void init(const std::shared_ptr<SceneGraph> &scene_graph);

            virtual void init_on_device() {
                _sampler.allocate_device(1);
            }

            LM_NODISCARD const SceneData *scene_data() const;

            LM_NODISCARD virtual int spp() const { return _sampler->spp(); }

            LM_NODISCARD virtual uint frame_index() const = 0;

            LM_NODISCARD virtual Sensor *camera();

            LM_NODISCARD uint2 resolution() const;

            virtual void update() = 0;

            virtual void render() = 0;
        };

        class GPUIntegrator : public Integrator {

        public:
            GPUIntegrator(Device *device, Context *context)
                    : Integrator(device, context) {}
        };
    }
}