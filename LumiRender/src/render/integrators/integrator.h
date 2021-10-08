//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "render/scene/scene_graph.h"
#include "render/sensors/sensor.h"
#include "core/backend/synchronizer.h"
#include "render/samplers/sampler.h"

// todo compile error
//#include "render/include/scene_data.h"


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
            Managed<Sensor, Sensor> _camera{_device};
        public:
            Integrator(Device *device, Context *context)
                    : _device(device),
                      _context(context) {}

            virtual ~Integrator() = default;

            virtual void init(const std::shared_ptr<SceneGraph> &scene_graph);

            virtual void init_on_device() {
                _camera.allocate_device(1);
                _sampler.allocate_device(1);
            }

            LM_NODISCARD const SceneData *scene_data() const;

            LM_NODISCARD virtual int spp() const { return _sampler->spp(); }

            LM_NODISCARD virtual uint frame_index() const = 0;

            LM_NODISCARD virtual Sensor *camera() { return _camera.data(); }

            LM_NODISCARD uint2 resolution() const { return _camera->resolution(); }

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