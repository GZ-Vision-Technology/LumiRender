//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "parser/scene_graph.h"
#include "core/backend/synchronizer.h"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"

namespace luminous {

    inline namespace utility {
    class ProgressReporter;
    };

    inline namespace render {

        class Scene;

        class Integrator : public Noncopyable {
        protected:
            uint _max_depth{};
            uint _min_depth{};
            float _rr_threshold{};
            Device *_device{};
            SP<Scene> _scene{};
            Context *_context{};
            Managed<Sampler, Sampler> _sampler{_device};
            Synchronizer<Sensor> _camera{_device};
            Dispatcher *_dispatcher{};
        public:
            uint2 debug_pixel{};

            Integrator(Device *device, Context *context)
                    : _device(device),
                      _context(context),
                      _dispatcher(_device->get_dispatcher()) {}

            virtual ~Integrator() = default;

            virtual void init(const std::shared_ptr<SceneGraph> &scene_graph);

            virtual void init_on_device() {
                _sampler.allocate_device(1);
                _sampler.synchronize_to_device();
            }

            LM_NODISCARD const SceneData *scene_data() const;

            LM_NODISCARD virtual int spp() const { return _sampler->spp(); }

            LM_NODISCARD virtual uint frame_index() const = 0;

            LM_NODISCARD virtual Sensor *camera();

            virtual void update_resolution(uint2 resolution) {}

            LM_NODISCARD uint2 resolution() const;

            virtual void update() = 0;

            virtual void render(int frame_num, ProgressReporter *progressor) = 0;
        };
    }
}